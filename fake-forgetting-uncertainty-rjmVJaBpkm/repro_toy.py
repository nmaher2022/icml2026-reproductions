#!/usr/bin/env python3
"""
CPU toy reproduction of ICML 2026 #2446
"Tackling Fake Forgetting through Uncertainty Quantification" (arXiv 2501.19403).

Faithfully reuses the official repo code (Conformal-Prediction-Unlearning/):
  - models.resnet.ResNet18
  - unlearn.py  (retrain, finetune, RL, GA, ga_plus[=NegGrad+], teacher, ssd, salun)
  - unlearn_cpu.py (CPU framework: L = L_original + lamda * clamp(q_hat-(1-p_true)+delta,0))
  - metrics.CR / metrics.MIACR

Scaled down for a CPU-only machine (no GPU, HF Jobs blocked by 402). Everything is
parameterized so the SAME script re-runs at full scale on a GPU (set --n_per_class 5000
--device cuda --orig_epochs 200 etc.).

Outputs (under --outdir):
  ckpt/original.pth, ckpt/<method>.pth        model checkpoints (resumable: skipped if present)
  results.json                                all metrics per method
  logs/*.txt                                  per-step console captured by trackio logbook run

Metrics per model (mirrors paper Tables 1-4,6):
  UA = 1-acc(forget), RA = acc(retain), TA = acc(test), MIA (traditional SVC), MIACR
  Coverage/SetSize/CR on forget (Df) and test (Dtest)
  Fake forgetting (Table 2): among forget samples MISCLASSIFIED (argmax!=GT),
    fraction whose GT label is still inside the conformal prediction set.
"""
import argparse, os, sys, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Conformal-Prediction-Unlearning")
sys.path.insert(0, REPO)

# ---- stub wandb (unlearn.py calls wandb.log internally; we don't want the dependency) ----
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
import types
_wandb = types.ModuleType("wandb")
_wandb.init = lambda *a, **k: None
_wandb.log = lambda *a, **k: None
_wandb.watch = lambda *a, **k: None
_wandb.save = lambda *a, **k: None
_wandb.config = types.SimpleNamespace(update=lambda *a, **k: None)
_wandb.finish = lambda *a, **k: None
sys.modules["wandb"] = _wandb

import utils
import unlearn
import unlearn_cpu
from metrics import CR, MIACR
from models import resnet
import torchvision
import torchvision.transforms as T


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

def _train_tf():
    # LIGHT augmentation for the CPU toy: on a 2.5k-image subset the repo's full augmentation
    # (RandomCrop + RandomGrayscale) blocks the model from fitting within a CPU-feasible number of
    # gradient steps -> underfit -> the fake-forgetting premise (model has LEARNED the data) breaks.
    # Flip-only lets the model both fit the train set and keep some generalization (so the held-out
    # conformal calibration is meaningful). The GPU full-scale re-run uses the repo's full augmentation.
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def _test_tf():
    return T.Compose([T.Resize(32), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def build_data(args):
    root = args.data_dir
    train_full = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=_train_tf())
    train_full_eval = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=_test_tf())
    test_full = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=_test_tf())

    # subsample n_per_class per class (deterministic)
    targets = np.array(train_full.targets)
    rng = np.random.RandomState(args.seed)
    sel = []
    for c in range(10):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        sel.extend(idx_c[: args.n_per_class].tolist())
    sel = np.array(sel)
    rng.shuffle(sel)

    # random forget/retain split
    n_forget = int(round(len(sel) * args.forget_ratio))
    forget_idx = sel[:n_forget]
    retain_idx = sel[n_forget:]

    # training loaders (train transform, with augmentation) -- used by repo unlearn fns
    forget_ds = Subset(train_full, forget_idx.tolist())
    retain_ds = Subset(train_full, retain_idx.tolist())
    # eval loaders (test transform, no augmentation) -- used for stable metrics
    forget_ds_ev = Subset(train_full_eval, forget_idx.tolist())
    retain_ds_ev = Subset(train_full_eval, retain_idx.tolist())

    # test set: split off a 'cal_size' calibration set (paper: 2000) + test remainder
    n_test = len(test_full)
    g = torch.Generator().manual_seed(args.seed)
    test_ds, cal_ds = random_split(test_full, [n_test - args.cal_size, args.cal_size], generator=g)

    bs = args.batch_size
    loaders = dict(
        train_forget=DataLoader(forget_ds, batch_size=bs, shuffle=True, num_workers=args.workers),
        train_retain=DataLoader(retain_ds, batch_size=bs, shuffle=True, num_workers=args.workers),
        forget_ev=DataLoader(forget_ds_ev, batch_size=bs, shuffle=False, num_workers=args.workers),
        retain_ev=DataLoader(retain_ds_ev, batch_size=bs, shuffle=False, num_workers=args.workers),
        test=DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=args.workers),
        cal=DataLoader(cal_ds, batch_size=bs, shuffle=False, num_workers=args.workers),
        # small test loader for fast per-epoch logging inside repo unlearn fns
        test_small=DataLoader(Subset(test_ds, list(range(min(1000, len(test_ds))))),
                              batch_size=bs, shuffle=False, num_workers=args.workers),
        # capped test subset for the O(n^2) SVC-based MIA / MIACR (keeps eval fast)
        test_mia=DataLoader(Subset(test_ds, list(range(min(args.mia_cap, len(test_ds))))),
                            batch_size=bs, shuffle=False, num_workers=args.workers),
        full_train=DataLoader(ConcatDataset((retain_ds, forget_ds)), batch_size=bs, num_workers=args.workers),
    )
    meta = dict(n_train=len(sel), n_forget=len(forget_idx), n_retain=len(retain_idx),
                n_test=len(test_ds), n_cal=len(cal_ds))
    return loaders, meta


# --------------------------------------------------------------------------------------
# Original training
# --------------------------------------------------------------------------------------
def train_original(args, loaders, device, path):
    net = resnet.ResNet18(num_classes=10).to(device)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
        print(f"[original] loaded existing checkpoint {path}")
        return net
    # train on full subsampled train (retain+forget) -- this is the "original model theta_o"
    train_loader = loaders["full_train"]
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.orig_epochs)
    for ep in range(args.orig_epochs):
        net.train(); t0 = time.time(); tot = 0; corr = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out = net(x); loss = crit(out, y)
            loss.backward(); opt.step()
            corr += (out.argmax(1) == y).sum().item(); tot += y.numel()
        sched.step()
        if (ep + 1) % max(1, args.orig_epochs // 10) == 0 or ep == args.orig_epochs - 1:
            ta = utils.evaluate_acc(net, loaders["test_small"], device)
            print(f"[original] ep {ep+1}/{args.orig_epochs} train_acc={corr/tot:.4f} test_acc={ta:.4f} ({time.time()-t0:.0f}s)")
    torch.save(net.state_dict(), path)
    print(f"[original] saved {path}")
    return net


# --------------------------------------------------------------------------------------
# Unlearning dispatch (reuses repo functions)
# --------------------------------------------------------------------------------------
class _Args:  # lightweight stand-in for argparse namespace expected by some repo fns
    pass

def make_kwargs(args, loaders, net, device, method):
    a = _Args()
    a.num_epochs = args.unlearn_epochs
    a.batch_size = args.batch_size
    kwargs = dict(
        model=net,
        train_retain_dl=loaders["train_retain"],
        train_forget_dl=loaders["train_forget"],
        test_retain_dl=loaders["test_small"],   # per-epoch logging only
        test_forget_dl=loaders["test_small"],
        num_classes=10,
        device=device,
        model_name="resnet18",
        num_epochs=args.unlearn_epochs,
        learning_rate=args.unlearn_lr,
        milestones=None,
        batch_size=args.batch_size,
        full_train_dl=loaders["full_train"],
        unlearning_teacher=resnet.ResNet18(num_classes=10).to(device) if method == "teacher" else None,
        unlearn_type="random",
        forget_class=-1,
        dampening_constant=1,
        selection_weighting=10,
        mask=None,
        args=a,
        save_dir=args.outdir,
    )
    if method == "retrain":
        kwargs["num_epochs"] = args.rt_epochs
        a.num_epochs = args.rt_epochs
        kwargs["learning_rate"] = args.learning_rate
    if method == "finetune":
        kwargs["learning_rate"] = args.ft_lr
    if method == "GA":
        kwargs["learning_rate"] = args.ga_lr   # GA (gradient ascent) needs a small lr or it diverges to NaN
    if method == "salun":
        kwargs["mask"] = build_salun_mask(net, loaders, device, args)
    return kwargs

def build_salun_mask(net, loaders, device, args, threshold=0.5):
    # SalUn saliency mask: |grad| of loss on forget data, top-`threshold` fraction per tensor
    net = copy.deepcopy(net).to(device)
    crit = nn.CrossEntropyLoss()
    net.eval()
    grads = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    for x, y in loaders["train_forget"]:
        x, y = x.to(device), y.to(device)
        net.zero_grad()
        loss = -crit(net(x), y)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                grads[n] += p.grad.detach().abs()
    mask = {}
    for n, g in grads.items():
        flat = g.flatten()
        k = int(len(flat) * threshold)
        if k < 1:
            mask[n] = torch.ones_like(g)
            continue
        thr = torch.sort(flat)[0][-k]
        mask[n] = (g >= thr).float()
    return mask

def run_method(args, loaders, orig_net, device, method):
    net = copy.deepcopy(orig_net).to(device)
    if method == "retrain":
        net = resnet.ResNet18(num_classes=10).to(device)  # retrain from scratch on retain only
    kwargs = make_kwargs(args, loaders, net, device, method)
    fn = getattr(unlearn, method)
    t0 = time.time()
    out = fn(**kwargs)
    if isinstance(out, tuple) and len(out) == 4:  # sfron returns model too
        net = out[3]
    print(f"[{method}] done in {time.time()-t0:.0f}s")
    return net


def run_cpu_method(args, loaders, orig_net, device, base_method, lamda):
    """CPU framework via unlearn_cpu.py (finetune/RL/retrain)."""
    net = copy.deepcopy(orig_net).to(device)
    if base_method == "retrain":
        net = resnet.ResNet18(num_classes=10).to(device)
    a = _Args(); a.num_epochs = args.unlearn_epochs; a.batch_size = args.batch_size
    # CPU needs a calibration loader drawn from test-retain (see main_unlearn_cpu.py)
    kwargs = dict(
        model=net,
        train_retain_dl=loaders["train_retain"],
        train_forget_dl=loaders["train_forget"],
        test_retain_dl=loaders["test_small"],
        test_forget_dl=loaders["test_small"],
        cal_dl=loaders["cal"],
        num_classes=10,
        device=device,
        model_name="resnet18",
        num_epochs=args.unlearn_epochs,
        learning_rate=args.ft_lr if base_method == "finetune" else args.unlearn_lr,
        milestones=None,
        batch_size=args.batch_size,
        full_train_dl=loaders["full_train"],
        unlearning_teacher=None,
        delta=args.delta,
        alpha=args.cpu_alpha,
        lamda=lamda,
        unlearn_type="random",
        unlearn_name=base_method,
        mask=None,
        args=a,
    )
    if base_method == "retrain":
        kwargs["num_epochs"] = args.rt_epochs
        kwargs["learning_rate"] = args.learning_rate
    fn = getattr(unlearn_cpu, base_method)
    t0 = time.time()
    fn(**kwargs)
    print(f"[cpu-{base_method} lamda={lamda}] done in {time.time()-t0:.0f}s")
    return net


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_probs(net, loader, device):
    net.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = F.softmax(net(x), dim=1)
        probs.append(p.cpu()); labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()

def conformal_stats(probs, labels, q_hat):
    """coverage, set_size, CR + fake-forgetting counts, at a given q_hat."""
    pred = probs.argmax(1)
    sets = probs >= (1 - q_hat)                       # boolean [N, C]
    n = len(labels)
    in_set_true = sets[np.arange(n), labels]          # GT label inside set?
    coverage = in_set_true.mean()
    set_size = sets.sum(1).mean()
    cr = coverage / set_size if set_size > 0 else 0.0
    mis = pred != labels                              # UA-misclassified
    n_mis = int(mis.sum())
    in_set_among_mis = int(in_set_true[mis].sum())    # GT still recoverable by CP
    ratio = in_set_among_mis / n_mis if n_mis > 0 else 0.0
    return dict(coverage=float(coverage), set_size=float(set_size), cr=float(cr),
                n_mislabel=n_mis, n_in_set=in_set_among_mis, recover_ratio=float(ratio))

def calibrate_qhat(probs_cal, labels_cal, alpha):
    n = len(labels_cal)
    scores = 1 - probs_cal[np.arange(n), labels_cal]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, q_level, method="higher"))

def traditional_mia(loaders, net, device):
    """confidence-based SVC MIA efficacy: predict membership on forget set."""
    # shadow: retain(train) vs test; target: forget(train) vs test
    st_p, st_y = MIACR.collect_prob(loaders["retain_ev"], net, device)
    se_p, se_y = MIACR.collect_prob(loaders["test_mia"], net, device)
    tt_p, tt_y = MIACR.collect_prob(loaders["forget_ev"], net, device)
    st_conf = torch.gather(st_p, 1, st_y[:, None])
    se_conf = torch.gather(se_p, 1, se_y[:, None])
    tt_conf = torch.gather(tt_p, 1, tt_y[:, None])
    # SVC_fit_predict returns mean predicted-membership on target(train) => higher = looks like member
    acc = MIACR.SVC_fit_predict(st_conf, se_conf, tt_conf, torch.zeros([0, 1]))
    return float(acc)

def miacr(loaders, net, device):
    """conformal MIA (MIACR) via repo SVC_MIA -> returns tuple; expose the CR-like number."""
    try:
        out = MIACR.SVC_MIA(
            shadow_train=loaders["retain_ev"],
            shadow_test=loaders["test_mia"],
            target_train=loaders["forget_ev"],
            target_test=loaders["test_mia"],
            cal_dl=loaders["cal"],
            model=net, device=device,
        )
        # out = (acc_train, cov_train, size_train, q_hat[, acc_test, cov_test, size_test])
        acc_tr, cov_tr, size_tr = float(out[0]), float(out[1]), float(out[2])
        cr = cov_tr / size_tr if size_tr > 0 else 0.0
        return dict(mia_acc=acc_tr, mia_coverage=cov_tr, mia_set_size=size_tr, miacr=float(cr))
    except Exception as e:
        return dict(error=str(e))

def evaluate(net, loaders, device, alphas):
    net.eval()
    pf, yf = collect_probs(net, loaders["forget_ev"], device)
    pr, yr = collect_probs(net, loaders["retain_ev"], device)
    pt, yt = collect_probs(net, loaders["test"], device)
    pc, yc = collect_probs(net, loaders["cal"], device)
    forget_acc = float((pf.argmax(1) == yf).mean())
    res = dict(
        UA=float(1 - forget_acc), forget_acc=forget_acc,
        RA=float((pr.argmax(1) == yr).mean()),
        TA=float((pt.argmax(1) == yt).mean()),
        per_alpha={},
    )
    for alpha in alphas:
        q = calibrate_qhat(pc, yc, alpha)
        res["per_alpha"][f"{alpha}"] = dict(
            q_hat=q,
            forget=conformal_stats(pf, yf, q),
            test=conformal_stats(pt, yt, q),
        )
    res["MIA_traditional"] = traditional_mia(loaders, net, device)
    res["MIACR"] = miacr(loaders, net, device)
    return res


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--outdir", default="./outputs")
    ap.add_argument("--n_per_class", type=int, default=300)
    ap.add_argument("--forget_ratio", type=float, default=0.1)
    ap.add_argument("--cal_size", type=int, default=2000)
    ap.add_argument("--mia_cap", type=int, default=2000, help="cap on test subset for SVC MIA/MIACR (speed)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--orig_epochs", type=int, default=25)
    ap.add_argument("--rt_epochs", type=int, default=25)
    ap.add_argument("--unlearn_epochs", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.1)   # original / retrain
    ap.add_argument("--unlearn_lr", type=float, default=0.01)     # RL/salun/ga_plus
    ap.add_argument("--ft_lr", type=float, default=0.01)          # finetune
    ap.add_argument("--ga_lr", type=float, default=1e-4)          # gradient ascent (small to avoid NaN)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--cpu_alpha", type=float, default=0.05)
    ap.add_argument("--lamda", type=float, default=0.5)
    ap.add_argument("--alphas", default="0.05,0.1")
    ap.add_argument("--methods", default="retrain,finetune,RL,GA,ga_plus,teacher,ssd,salun")
    ap.add_argument("--cpu_methods", default="finetune,RL")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    utils.setup_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(os.path.join(args.outdir, "ckpt"), exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    results_path = os.path.join(args.outdir, "results.json")
    results = {}
    if os.path.exists(results_path):
        results = json.load(open(results_path))

    loaders, meta = build_data(args)
    results["_meta"] = dict(meta, args=vars(args))
    json.dump(results, open(results_path, "w"), indent=2)
    print("META:", meta)

    orig_path = os.path.join(args.outdir, "ckpt", "original.pth")
    orig = train_original(args, loaders, device, orig_path)
    if "original" not in results:
        utils.setup_seed(args.seed)
        results["original"] = evaluate(orig, loaders, device, alphas)
        json.dump(results, open(results_path, "w"), indent=2)
        print("ORIGINAL:", json.dumps(results["original"], indent=2))

    methods = [m for m in args.methods.split(",") if m]
    for method in methods:
        if method in results:
            print(f"[{method}] already in results, skipping"); continue
        ck = os.path.join(args.outdir, "ckpt", f"{method}.pth")
        try:
            utils.setup_seed(args.seed)
            if os.path.exists(ck):
                net = (resnet.ResNet18(num_classes=10).to(device))
                net.load_state_dict(torch.load(ck, map_location=device))
            else:
                net = run_method(args, loaders, orig, device, method)
                torch.save(net.state_dict(), ck)
            utils.setup_seed(args.seed)
            results[method] = evaluate(net, loaders, device, alphas)
            json.dump(results, open(results_path, "w"), indent=2)
            print(f"{method.upper()}:", json.dumps(results[method], indent=2))
        except Exception as e:
            import traceback; traceback.print_exc()
            results[method] = {"error": f"{type(e).__name__}: {e}"}
            json.dump(results, open(results_path, "w"), indent=2)
            print(f"[{method}] FAILED -> recorded error, continuing")

    for base in [m for m in args.cpu_methods.split(",") if m]:
        for lamda in [0.0, args.lamda]:
            tag = f"cpu-{base}-l{lamda}"
            if tag in results:
                print(f"[{tag}] already in results, skipping"); continue
            ck = os.path.join(args.outdir, "ckpt", f"{tag}.pth")
            try:
                utils.setup_seed(args.seed)
                if os.path.exists(ck):
                    net = resnet.ResNet18(num_classes=10).to(device)
                    net.load_state_dict(torch.load(ck, map_location=device))
                else:
                    net = run_cpu_method(args, loaders, orig, device, base, lamda)
                    torch.save(net.state_dict(), ck)
                utils.setup_seed(args.seed)
                results[tag] = evaluate(net, loaders, device, alphas)
                json.dump(results, open(results_path, "w"), indent=2)
                print(f"{tag.upper()}:", json.dumps(results[tag], indent=2))
            except Exception as e:
                import traceback; traceback.print_exc()
                results[tag] = {"error": f"{type(e).__name__}: {e}"}
                json.dump(results, open(results_path, "w"), indent=2)
                print(f"[{tag}] FAILED -> recorded error, continuing")

    print("ALL DONE. results ->", results_path)


if __name__ == "__main__":
    main()
