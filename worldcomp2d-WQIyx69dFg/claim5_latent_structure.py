"""Claim 5: structured latent space (Fig 5a-c), using the pretrained COFW PdEnc directly on real
COFW test images -- no training needed for these three panels (Fig 5d, the proximity-unweighted
ablation encoder, is a from-scratch training run and handled separately/not attempted here, see
BUGFIX_LOG.md).

Landmark index note: `Utils.fliplr_joints`'s COFW mirror-pairing (Test/Utils.py:28-29) includes
pair [17, 18] (1-indexed) with no other unpaired pupil-like candidates -- this is the standard
COFW-29 left/right pupil pair, confirming the paper's "left pupil (Class 17)" is 1-indexed landmark
17 = 0-indexed index 16 in this codebase's point ordering. Used directly, not re-derived.

(a) Intra-class clustering (Fig 5a): for each of 29 landmark classes, extract PdEnc(o) at the
    ground-truth landmark position across N test images, compute the per-class mean embedding, and
    report each instance's L2 distance to its class mean -- expect small intra-class spread
    relative to inter-class centroid separation ("successful class-wise clustering").
(b) Inter-class separation (Fig 5b): within each image, L2 distance from the left-pupil (class 16)
    embedding to every other class's embedding, averaged over images; correlated (Spearman)
    against real-world pixel distance between the same landmark pairs -- expect a positive
    correlation ("spatially adjacent landmarks ... exhibit relatively smaller distances").
(c) Distance-preserving embedding (Fig 5c): fixation point anchored at the left-pupil landmark;
    query points sampled at controlled real-world pixel offsets (multiple random directions per
    radius) from the anchor; L2 latent distance from the anchor embedding vs. offset radius --
    expect monotonic increase within RF[2]'s ~54px range (per Sec 4.2).

Run with: <repo-root>/.venv/bin/python worldcomp2d-WQIyx69dFg/claim5_latent_structure.py
"""
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
WC2D_TEST = os.path.join(REPO_ROOT, "WorldComp2D", "Test")
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) + "/"

sys.path.insert(0, WC2D_TEST)
os.chdir(WC2D_TEST)

import Utils as U  # noqa: E402
from Models import framework_making  # noqa: E402

device = torch.device("cpu")
LEFT_PUPIL = 16  # 0-indexed; paper's "Class 17" (1-indexed)
N_IMAGES = 150   # subsample of the 507-image test set, for speed


def get_embeddings():
    """Returns Z: (N_IMAGES, 29, 256) PdEnc embeddings at ground-truth landmark positions,
    and P: (N_IMAGES, 29, 2) their (y, x) positions in 256x256 crop space."""
    ds = U.COFW(DATA_PATH, is_train=False, random_scale=False, random_flip=False, random_rotation=False)
    encoders, _, _, n_l, _ = framework_making("COFW")
    encoder = encoders[0].eval()  # any of the 3 trained instances; consistent throughout
    assert n_l == 29

    Z, P = [], []
    with torch.no_grad():
        for i in range(N_IMAGES):
            img, tpts, pts, center, scale = ds[i]
            pts_yx = tpts.view(29, 2).float()  # already (y, x) order, 256x256 crop space (see docstring)
            o = U.extract_observation("COFW", img.unsqueeze(0), pts_yx.long())
            z = encoder(o)  # (29, 256)
            Z.append(z.numpy())
            P.append(pts_yx.numpy())
    return np.stack(Z), np.stack(P)


def analyze_intra_inter_class(Z):
    N, n_l, _ = Z.shape
    class_means = Z.mean(axis=0)  # (29, 256)
    intra = np.array([np.linalg.norm(Z[:, c, :] - class_means[c], axis=1).mean() for c in range(n_l)])

    inter_pairs = []
    for c1 in range(n_l):
        for c2 in range(c1 + 1, n_l):
            inter_pairs.append(np.linalg.norm(class_means[c1] - class_means[c2]))
    inter_pairs = np.array(inter_pairs)

    print("(a) Intra-class clustering:")
    print(f"    mean intra-class L2 (instance-to-class-mean): {intra.mean():.4f} +/- {intra.std():.4f}")
    print(f"    mean inter-class L2 (class-mean-to-class-mean): {inter_pairs.mean():.4f} +/- {inter_pairs.std():.4f}")
    ratio = inter_pairs.mean() / intra.mean()
    print(f"    inter/intra ratio: {ratio:.2f}x "
          f"({'clear class-wise clustering' if ratio > 2 else 'weak/no clear clustering'})")
    return intra, class_means


def analyze_inter_class_separation(Z, P):
    N, n_l, _ = Z.shape
    latent_dist = np.linalg.norm(Z[:, LEFT_PUPIL:LEFT_PUPIL + 1, :] - Z, axis=2).mean(axis=0)  # (29,)
    realworld_dist = np.linalg.norm(P[:, LEFT_PUPIL:LEFT_PUPIL + 1, :] - P, axis=2).mean(axis=0)  # (29,)

    mask = np.arange(n_l) != LEFT_PUPIL
    rho, pval = spearmanr(realworld_dist[mask], latent_dist[mask])
    print("\n(b) Inter-class separation (left pupil, class 16, vs. all other classes):")
    order = np.argsort(realworld_dist[mask])
    classes = np.arange(n_l)[mask]
    nearest3 = classes[order[:3]]
    farthest3 = classes[order[-3:]]
    print(f"    nearest-3 classes (real-world) latent dist: {latent_dist[nearest3]}")
    print(f"    farthest-3 classes (real-world) latent dist: {latent_dist[farthest3]}")
    print(f"    Spearman rho(real-world dist, latent dist) across 28 other classes: {rho:.3f} (p={pval:.4f})")
    print(f"    {'positive correlation -- spatially adjacent landmarks are latently closer, as claimed' if rho > 0 else 'NO positive correlation -- claim not supported'}")
    return rho


def analyze_distance_preservation(radii=(5, 10, 15, 20, 27, 35, 45, 54, 70, 90, 110, 130), n_dirs=8, n_images=60):
    ds = U.COFW(DATA_PATH, is_train=False, random_scale=False, random_flip=False, random_rotation=False)
    encoders, _, _, n_l, _ = framework_making("COFW")
    encoder = encoders[0].eval()

    rng = np.random.RandomState(0)
    per_radius = {r: [] for r in radii}
    with torch.no_grad():
        for i in range(n_images):
            img, tpts, pts, center, scale = ds[i]
            pts_yx = tpts.view(29, 2).float()
            anchor_yx = pts_yx[LEFT_PUPIL]

            points = [anchor_yx.tolist()]
            meta = [("anchor", 0)]
            for r in radii:
                for d in range(n_dirs):
                    theta = 2 * np.pi * d / n_dirs
                    dy, dx = r * np.sin(theta), r * np.cos(theta)
                    y = float(np.clip(anchor_yx[0].item() + dy, 0, 255))
                    x = float(np.clip(anchor_yx[1].item() + dx, 0, 255))
                    points.append([y, x])
                    meta.append((r, d))

            pts_batch = torch.tensor(points).long()
            o = U.extract_observation("COFW", img.unsqueeze(0), pts_batch)
            z = encoder(o)  # (1 + len(radii)*n_dirs, 256)
            z_anchor = z[0]
            for j, (r, d) in enumerate(meta[1:], start=1):
                dist = torch.norm(z[j] - z_anchor).item()
                per_radius[r].append(dist)

    radii_arr = np.array(radii)
    mean_dist = np.array([np.mean(per_radius[r]) for r in radii])
    rho, pval = spearmanr(radii_arr, mean_dist)
    print(f"\n(c) Distance-preserving embedding (anchor = left pupil, n={n_images} images x {n_dirs} directions):")
    for r, m in zip(radii, mean_dist):
        flag = " <= RF[2] (~54px)" if r <= 54 else ""
        print(f"    r={r:4d}px  mean latent L2 = {m:.4f}{flag}")
    within_mask = radii_arr <= 54
    rho_within, p_within = spearmanr(radii_arr[within_mask], mean_dist[within_mask])
    print(f"    Spearman rho (all radii): {rho:.3f} (p={pval:.4f})")
    print(f"    Spearman rho (radii <= 54px, RF[2] range): {rho_within:.3f} (p={p_within:.4f})")
    print(f"    {'monotonic increase within RF[2], as claimed' if rho_within > 0.9 else 'NOT cleanly monotonic within RF[2]'}")
    return rho_within


def main():
    print(f"Extracting embeddings for {N_IMAGES} COFW test images...")
    Z, P = get_embeddings()
    analyze_intra_inter_class(Z)
    analyze_inter_class_separation(Z, P)
    analyze_distance_preservation()


if __name__ == "__main__":
    main()
