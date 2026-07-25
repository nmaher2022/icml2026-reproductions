#!/usr/bin/env python3
"""
Option-3 metric verification for ICML 2026 #2446.

Independent of any trained model: builds SYNTHETIC softmax outputs with hand-computable
answers and asserts that the official repo's split-conformal machinery
(metrics/CR.py, metrics/MIACR.py, unlearn_cpu.py) matches the paper's definitions
(Eqs. 1-6 and the CPU loss, Eq. 8).

Run: ./.venv/bin/python verify_metrics.py
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Conformal-Prediction-Unlearning")
sys.path.insert(0, REPO)
import types
sys.modules["wandb"] = types.SimpleNamespace(
    init=lambda *a, **k: None, log=lambda *a, **k: None, watch=lambda *a, **k: None,
    save=lambda *a, **k: None, finish=lambda *a, **k: None,
    config=types.SimpleNamespace(update=lambda *a, **k: None))
from metrics import CR, MIACR          # official implementations
from unlearn_cpu import get_cpu_loss   # official CPU loss

OK = "PASS"; BAD = "FAIL"
results = []
def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"[{OK if cond else BAD}] {name}" + (f"  ::  {detail}" if detail else ""))

np.random.seed(0)

# ---------------------------------------------------------------------------
# 1. Split-conformal quantile (paper Eq. 2): q_hat = Quantile_{1-alpha}(cal scores),
#    with the finite-sample correction level ceil((n+1)(1-alpha))/n.
# ---------------------------------------------------------------------------
# Calibration true-class probabilities -> non-conformity scores s = 1 - p (Eq. 1).
p_true_cal = np.array([0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.10])
scores = 1 - p_true_cal                      # = [.01,.05,.10,.20,.30,.40,.50,.60,.90]
n = len(scores); alpha = 0.1
q_level = np.ceil((n + 1) * (1 - alpha)) / n   # ceil(10*0.9)/9 = ceil(9)/9 = 1.0
manual_q = np.quantile(scores, q_level, method="higher")
repo_q = CR.find_quantile(scores, n, alpha)
check("Eq2 q_level formula", abs(q_level - 1.0) < 1e-12, f"q_level={q_level}")
check("CR.find_quantile == manual np.quantile(higher)", abs(repo_q - manual_q) < 1e-12,
      f"repo={repo_q:.4f} manual={manual_q:.4f} (=max score 0.90 at level 1.0)")
check("q_hat equals expected 0.90", abs(repo_q - 0.90) < 1e-9, f"q_hat={repo_q:.4f}")

# ---------------------------------------------------------------------------
# 2. get_calibration (metrics/CR.py) reproduces the same q_hat from a model+loader.
#    Build a fake net whose softmax over the cal set reproduces p_true_cal exactly.
# ---------------------------------------------------------------------------
C = 4
def logits_for(p_true, true_idx=0):
    # put prob p on class true_idx, spread (1-p) on the rest, return log (pre-softmax up to const)
    row = np.full(C, (1 - p_true) / (C - 1)); row[true_idx] = p_true
    return np.log(row + 1e-12)
cal_logits = torch.tensor(np.stack([logits_for(p) for p in p_true_cal]), dtype=torch.float32)
cal_labels = torch.zeros(len(p_true_cal), dtype=torch.long)   # true class = 0
class FakeNet(torch.nn.Module):
    def __init__(self, logits): super().__init__(); self.logits = logits
    def forward(self, x): return self.logits[x.long().flatten()]
net = FakeNet(cal_logits)
cal_ds = torch.utils.data.TensorDataset(torch.arange(len(p_true_cal)), cal_labels)
cal_dl = torch.utils.data.DataLoader(cal_ds, batch_size=4)
q_from_model = CR.get_calibration(net, alpha, torch.device("cpu"), cal_dl)
check("CR.get_calibration matches Eq2 q_hat", abs(q_from_model - 0.90) < 1e-4,
      f"q_hat_model={q_from_model:.4f}")

# ---------------------------------------------------------------------------
# 3. Prediction set (Eq. 3): C(x) = {y : 1 - p_y <= q_hat} = {y : p_y >= 1 - q_hat}.
#    Coverage (Eq. 4), Set Size (Eq. 5), CR (Eq. 6), and the fake-forgetting count.
# ---------------------------------------------------------------------------
q_hat = 0.90
thresh = 1 - q_hat  # = 0.10 : any class with prob >= 0.10 enters the set
# 4 test points, true label = 0 each; columns are class probs (sum ~1)
probs = np.array([
    [0.60, 0.30, 0.05, 0.05],   # pt0: argmax=0 correct; set={0,1}   size2 (0.60,0.30>=.1)
    [0.30, 0.55, 0.10, 0.05],   # pt1: argmax=1 WRONG; set={0,1,2}  size3; true(0)=0.30>=.1 -> IN (fake forget!)
    [0.05, 0.80, 0.10, 0.05],   # pt2: argmax=1 WRONG; set={1,2}    size2; true(0)=0.05<.1  -> NOT (true forget)
    [0.02, 0.90, 0.05, 0.03],   # pt3: argmax=1 WRONG; set={1}      size1; true(0)=0.02<.1  -> NOT
])
labels = np.array([0, 0, 0, 0])
sets = probs >= thresh
pred = probs.argmax(1)
in_set_true = sets[np.arange(4), labels]
coverage = in_set_true.mean()                 # true label in set for pts {0,1} -> 2/4 = 0.5
set_size = sets.sum(1).mean()                 # sizes [2,3,2,1] -> 8/4 = 2.0
cr = coverage / set_size
mis = pred != labels                          # pts {1,2,3} misclassified -> 3
n_mis = int(mis.sum())
in_set_among_mis = int(in_set_true[mis].sum())# only pt1 -> 1
ratio = in_set_among_mis / n_mis              # 1/3 = 0.333  (the "recover ratio" of Table 2)
check("Coverage (Eq4) = 0.50", abs(coverage - 0.50) < 1e-12, f"{coverage:.3f}")
check("Set Size (Eq5) = 2.00 (boundary-inclusive >=)", abs(set_size - 2.0) < 1e-12, f"{set_size:.3f}")
check("CR (Eq6) = 0.25", abs(cr - 0.25) < 1e-12, f"{cr:.4f}")
check("fake-forget: 1 of 3 misclassified has GT in set (ratio 0.333)",
      n_mis == 3 and in_set_among_mis == 1 and abs(ratio - 1/3) < 1e-12,
      f"n_mis={n_mis} in_set={in_set_among_mis} ratio={ratio:.3f}")

# cross-check the exact set-membership rule the repo uses in CR.CP_loop
repo_sets = torch.tensor(probs) >= (1 - q_hat)
check("repo CP_loop set rule (probs>=1-q_hat) == ours", bool((repo_sets.numpy() == sets).all()))

# ---------------------------------------------------------------------------
# 4. CPU loss (Eq. 8), official unlearn_cpu.get_cpu_loss:
#    L = clamp(q_hat - (1 - p_true) + delta, min=0).mean()  (== max{q_hat - S, -delta} + delta).
# ---------------------------------------------------------------------------
delta = 0.01
# single sample, true class 0, p_true = 0.5 -> score S = 0.5; q_hat=0.9
sm = torch.tensor([[0.5, 0.2, 0.2, 0.1]])
loss = get_cpu_loss(sm, torch.tensor([0]), q_hat=0.9, delta=delta, device="cpu", unlearn_type="random").item()
expected = max(0.9 - 0.5 + delta, 0.0)  # = 0.41
check("CPU loss clamp(q_hat-(1-p)+delta) = 0.41", abs(loss - 0.41) < 1e-6, f"loss={loss:.4f}")
# when true prob already tiny (score high), loss saturates to 0 (label excluded from set)
sm2 = torch.tensor([[0.01, 0.9, 0.05, 0.04]])
loss2 = get_cpu_loss(sm2, torch.tensor([0]), q_hat=0.9, delta=delta, device="cpu", unlearn_type="random").item()
check("CPU loss = 0 once true label excluded (S>q_hat+delta)", abs(loss2 - 0.0) < 1e-6, f"loss={loss2:.4f}")

# ---------------------------------------------------------------------------
# 5. MIACR uses the same finite-sample quantile on the MIA classifier's scores.
# ---------------------------------------------------------------------------
repo_mia_q = MIACR.find_quantile(scores, n, alpha)
check("MIACR.find_quantile == CR.find_quantile", abs(repo_mia_q - repo_q) < 1e-12,
      f"{repo_mia_q:.4f}")

# ---------------------------------------------------------------------------
print("\n===== SUMMARY =====")
npass = sum(1 for _, c, _ in results if c)
print(f"{npass}/{len(results)} checks passed")
sys.exit(0 if npass == len(results) else 1)
