"""Claim 4: zero-shot cross-dataset transfer. A WorldComp2D model *trained on 300W* (68 landmarks,
using the official pretrained 300W checkpoint -- no 300W data needed, 300W's own test set is
access-blocked, see PAPER_BRIEFING.md) is evaluated zero-shot on COFW-68 (Ghiasi & Fowlkes 2014's
68-landmark re-annotation of the same 507 COFW test images, from the public
`golnazghiasi/cofw68-benchmark` GitHub repo, cloned into this folder).

Index mapping COFW-68 annotation file "{i}_points.mat" (i=1..507, MATLAB 1-indexed) -> our
COFW_test_color.mat sample (i-1) (0-indexed) was verified geometrically: the landmark bounding
box of "1_points.mat" (x:[58,186] y:[9.8,118]) tightly matches sample index 0's 29-point COFW
annotation bbox (x:[65,158] y:[8.5,114]) -- same face, 68-point set is a superset extending to
jawline/contour points the 29-point set doesn't include.

Paper (Table 6): 300W (in-distribution) NME_IO = 5.06, COFW-68 (zero-shot) NME_IO = 6.08+-0.14,
degradation = 20.2%. We only have the pretrained 300W model + these COFW-68 images/annotations,
so we report the COFW-68 zero-shot number directly (the "5.06" in-distribution number is the same
as Claim 2's 300W NME, which is BLOCKED -- 300W's own test set/images are access-gated).

Run with: <repo-root>/.venv/bin/python worldcomp2d-WQIyx69dFg/claim4_cross_dataset.py
"""
import glob
import os
import sys

import h5py
import numpy as np
import torch
from scipy.io import loadmat

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
WC2D_TEST = os.path.join(REPO_ROOT, "WorldComp2D", "Test")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COFW_COLOR_MAT = os.path.join(THIS_DIR, "data_raw", "COFW_color", "COFW_test_color.mat")
COFW68_ANNOT_DIR = os.path.join(THIS_DIR, "cofw68-benchmark", "COFW68_Data", "test_annotations")

sys.path.insert(0, WC2D_TEST)
os.chdir(WC2D_TEST)

import Utils as U  # noqa: E402
from Models import framework_making  # noqa: E402

device = torch.device("cpu")
BATCH_SIZE = 39  # 507 / 39 = 13 exactly

FIXATION_POINTS = torch.LongTensor([[64, 64], [64, 128], [64, 192],
                                     [128, 64], [128, 128], [128, 192],
                                     [192, 64], [192, 128], [192, 192]]).to(device)


class COFW68(torch.utils.data.Dataset):
    """68-landmark COFW test images, mirrors IBUG_300W.__getitem__'s preprocessing exactly
    (same crop/normalize pipeline, since we're testing a 300W-trained model)."""

    def __init__(self):
        self.f = h5py.File(COFW_COLOR_MAT, "r")
        self.images = self.f["IsT"][0]
        n = len(self.images)
        assert n == 507, f"expected 507 COFW color test images, got {n}"
        self.annot_files = sorted(glob.glob(os.path.join(COFW68_ANNOT_DIR, "*_points.mat")),
                                   key=lambda p: int(os.path.basename(p).split("_")[0]))
        assert len(self.annot_files) == 507, f"expected 507 COFW-68 annotation files, got {len(self.annot_files)}"
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return 507

    def __getitem__(self, index):
        # annotation file "{index+1}_points.mat" (1-indexed) <-> our array position `index` (0-indexed)
        annot_path = os.path.join(COFW68_ANNOT_DIR, f"{index + 1}_points.mat")
        d = loadmat(annot_path)
        pts = d["Points"].astype(np.float32)  # (68, 2), x,y

        img_ref = self.images[index]
        # Raw h5py read of the MATLAB cell-array color image comes out axis-reversed
        # (like the official grayscale COFW loader's `np.array(img).T`, Test/Utils.py:165)
        # -- NOT a simple (C,H,W)->(H,W,C) channel move. Full reversal via .T is required.
        img = np.array(self.f[img_ref]).T.astype(np.float32)  # -> (H, W, C), or (W, H) if grayscale
        if img.ndim == 2:
            # 13/507 "color" mat entries are actually stored single-channel. Like the official
            # grayscale COFW loader, .T alone already yields (H, W) here -- just replicate to 3ch.
            img = np.stack([img, img, img], axis=-1)

        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        center_w = (np.floor(xmin) + np.ceil(xmax)) / 2.0
        center_h = (np.floor(ymin) + np.ceil(ymax)) / 2.0
        center = torch.Tensor([center_w, center_h])
        scale = max(np.ceil(xmax) - np.floor(xmin), np.ceil(ymax) - np.floor(ymin)) / 200.0
        scale *= 1.25

        img, scale_factor = U.crop(img, center, scale, [256, 256], rot=0)

        tpts = pts.copy()
        for i in range(pts.shape[0]):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = U.transform_pixel(tpts[i, 0:2] + 1, center, scale * scale_factor, [256, 256], rot=0)

        img = (img - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)

        tpts = np.fliplr(tpts).flatten().astype(np.float32)
        tpts = torch.LongTensor(tpts)
        pts_flat = torch.FloatTensor(pts.astype(np.float32)).flatten()

        return img, tpts, pts_flat, center, scale


def evaluate():
    ds = COFW68()
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

    task = "300W"  # use the 300W-trained checkpoint + 68-landmark/RGB conventions throughout
    encoders, localizers, aux_localizers, n_l, landmark_coordinate_prior = framework_making(task)
    for m in (encoders + localizers + aux_localizers):
        m.eval()
    assert n_l == 68

    fixation_points = FIXATION_POINTS.view(1, 9, 2).repeat(BATCH_SIZE, 1, 1)
    img_size = torch.FloatTensor([256, 256]).to(device)
    fixation_points_norm = (2 * fixation_points / (img_size - 1)) - 1
    l_idx = 2 * torch.arange(0, n_l).to(device) / (n_l - 1) - 1
    class_embedding = l_idx.view(-1, 1, 1, 1).expand(n_l, 1, 27, 27).repeat(BATCH_SIZE, 1, 1, 1)
    delta_clamp = 1  # non-COFW clamp value per Test.py

    NME_samples = [[] for _ in range(3)]
    with torch.no_grad():
        for images, tpts, pts, center, scale in loader:
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

    n_eval = len(NME_samples[0])
    print(f"Evaluated {n_eval} / 507 COFW-68 images (zero-shot, 300W-trained model).")
    NME_means = []
    for net_idx in range(3):
        m = torch.FloatTensor(NME_samples[net_idx]).mean().item() * 100
        NME_means.append(m)
        print(f"  model {net_idx + 1}: NME_IO (COFW-68, zero-shot) = {m:.4f}")
    NME_means_t = torch.FloatTensor(NME_means)
    ours_mean, ours_std = NME_means_t.mean().item(), NME_means_t.std().item()
    print(f"\nOurs:  COFW-68 zero-shot NME_IO = {ours_mean:.3f} +/- {ours_std:.3f}")
    print(f"Paper: COFW-68 zero-shot NME_IO = 6.08 +/- 0.14 (in-distribution 300W NME_IO = 5.06, "
          f"degradation 20.2% -- 300W in-distribution number not independently reproducible here, "
          f"300W's own test set is access-blocked, see PAPER_BRIEFING.md)")
    if ours_mean > 0:
        # degradation relative to paper's own in-distribution 300W figure (5.06), since we can't
        # independently reproduce that number ourselves without 300W test data
        degradation_vs_paper_indist = (ours_mean - 5.06) / 5.06 * 100
        print(f"Degradation vs paper's own in-distribution 300W NME (5.06): {degradation_vs_paper_indist:.1f}% "
              f"(paper's own reported degradation: 20.2%)")


if __name__ == "__main__":
    evaluate()
