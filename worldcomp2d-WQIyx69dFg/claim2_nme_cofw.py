"""Claim 2 (NME half) + basis for Claim 3/5: run the *official* eval pipeline (mirrors
Test/Test.py's test() function line-for-line) against the *real* COFW test set (507 images,
downloaded from data.caltech.edu, confirmed matching the paper's stated split size) using the
official pretrained COFW checkpoints. Only difference from Test/Test.py: data_path is passed
explicitly instead of relying on a hardcoded '../Dataset/' relative path, since our data lives in
this reproduction folder, not inside the vendored WorldComp2D/ clone.

Paper (Table 1/2, COFW): NME_IO = 5.16 +/- 0.05 (mean/std across the 3 independently-trained
model instances shipped in the checkpoint -- confirmed from Test.py: NME_samples has 3 entries,
mean/std computed *across those 3*, not across test images within one model).

Run with: <repo-root>/.venv/bin/python worldcomp2d-WQIyx69dFg/claim2_nme_cofw.py
"""
import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
WC2D_TEST = os.path.join(REPO_ROOT, "WorldComp2D", "Test")
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) + "/"

sys.path.insert(0, WC2D_TEST)
os.chdir(WC2D_TEST)  # load_proximity_dependent_encoder etc use "../Pretrained_modules/" relative paths

import Utils as U  # noqa: E402
from Models import framework_making  # noqa: E402

device = torch.device("cpu")
BATCH_SIZE = 39  # matches paper's Test.py default; 507 / 39 = 13 exactly, no drop_last loss

FIXATION_POINTS = torch.LongTensor([[64, 64], [64, 128], [64, 192],
                                     [128, 64], [128, 128], [128, 192],
                                     [192, 64], [192, 128], [192, 192]]).to(device)


def load_cofw_test_loader():
    test_set = U.COFW(DATA_PATH, is_train=False, random_scale=False, random_flip=False, random_rotation=False)
    assert len(test_set) == 507, f"expected 507 COFW test images, got {len(test_set)}"
    return torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                        num_workers=0, drop_last=True)


def evaluate(task="COFW"):
    test_loader = load_cofw_test_loader()
    encoders, localizers, aux_localizers, n_l, landmark_coordinate_prior = framework_making(task)
    for m in (encoders + localizers + aux_localizers):
        m.eval()

    fixation_points = FIXATION_POINTS.view(1, 9, 2).repeat(BATCH_SIZE, 1, 1)
    img_size = torch.FloatTensor([256, 256]).to(device)
    fixation_points_norm = (2 * fixation_points / (img_size - 1)) - 1
    l_idx = 2 * torch.arange(0, n_l).to(device) / (n_l - 1) - 1
    class_embedding = l_idx.view(-1, 1, 1, 1).expand(n_l, 1, 27, 27).repeat(BATCH_SIZE, 1, 1, 1)
    delta_clamp = 2 if task == "COFW" else 1

    NME_samples = [[] for _ in range(3)]
    n_batches = 0
    with torch.no_grad():
        for images, tpts, pts, center, scale in test_loader:
            images = images.to(device)
            o = U.extract_observation(task, images, fixation_points)

            for net_idx in range(3):
                z = encoders[net_idx](o)
                z_x = torch.cat((z, fixation_points_norm.view(-1, 2)), dim=1).view(
                    BATCH_SIZE, 9 * (z.size(1) + 2))
                x_hat = U.norm_coord_to_abs(landmark_coordinate_prior + localizers[net_idx](z_x)).long()

                o2 = U.extract_observation(task, images, x_hat, False)
                o_l = torch.cat((o2, class_embedding), dim=1)
                h = aux_localizers[net_idx](o_l)
                delta_y = torch.argmax(h.view(BATCH_SIZE, n_l, -1), dim=-1) // 27 - 13
                delta_x = torch.argmax(h.view(BATCH_SIZE, n_l, -1), dim=-1) % 27 - 13
                delta = torch.cat((delta_y.view(BATCH_SIZE, n_l, 1), delta_x.view(BATCH_SIZE, n_l, 1)), dim=-1)
                delta = torch.clamp(delta, -delta_clamp, delta_clamp)
                x_hat = x_hat + delta

                NME = U.NME_calc(task, x_hat.flip(dims=[-1]), pts, center, scale)
                NME_samples[net_idx].extend(NME.tolist())
            n_batches += 1

    n_evaluated = len(NME_samples[0])
    print(f"Evaluated {n_evaluated} images across {n_batches} batches (of 507 total; "
          f"{507 - n_evaluated} dropped by drop_last if batch size didn't divide evenly).")

    NME_means = []
    for net_idx in range(3):
        m = torch.FloatTensor(NME_samples[net_idx]).mean().item() * 100
        NME_means.append(m)
        print(f"  model {net_idx + 1}: NME_IO (COFW) = {m:.4f}")

    NME_means_t = torch.FloatTensor(NME_means)
    print(f"\nOurs:  NME_IO mean/std across 3 trained models = {NME_means_t.mean():.3f} +/- {NME_means_t.std():.3f}")
    print(f"Paper: NME_IO (COFW) = 5.16 +/- 0.05")


if __name__ == "__main__":
    evaluate("COFW")
