"""
2D hypervolume utilities for the MOCO reproduction (Claims 4-5).

Implements the paper's normalized hypervolume ratio
    HV'_r(F) = HV_r(F) / prod_i |r_i - z_i|
where r is a reference (nadir/worst) point and z is an ideal (best) point.
We do not have the paper's own reference/ideal points for our randomly
generated instances, so -- documented explicitly here and in the driver
script -- we define, per instance, r and z from the UNION of all methods'
observed raw objective values on that instance (elementwise worst / best
across every method's archive), separately for each objective's sense
(maximize for Knapsack profits, minimize for TSP tour lengths).
"""
from __future__ import annotations

import numpy as np


def _to_max_space(points, maximize):
    pts = np.asarray(points, dtype=float).copy()
    for j, mx in enumerate(maximize):
        if not mx:
            pts[:, j] = -pts[:, j]
    return pts


def pareto_mask_max(points_max):
    """Boolean mask of non-dominated points, all objectives maximized."""
    n = len(points_max)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        pi = points_max[i]
        for j in range(n):
            if i == j:
                continue
            pj = points_max[j]
            if np.all(pj >= pi) and np.any(pj > pi):
                keep[i] = False
                break
    return keep


def hv_2d_max(points_max, r_max):
    """Hypervolume dominated by non-dominated maximization points w.r.t.
    reference r_max (must be <= every point in every coordinate)."""
    if len(points_max) == 0:
        return 0.0
    order = np.argsort(-points_max[:, 0])
    pts = points_max[order]
    hv = 0.0
    prev_y = r_max[1]
    prev_x = r_max[0]
    for x, y in pts:
        if y > prev_y:
            hv += max(x - r_max[0], 0.0) * (y - prev_y)
            prev_y = y
        prev_x = x
    return hv


def nadir_ideal(points, maximize):
    """Per-objective worst (r) / best (z) point from a pool of raw points."""
    points = np.asarray(points, dtype=float)
    r = np.zeros(points.shape[1])
    z = np.zeros(points.shape[1])
    for j, mx in enumerate(maximize):
        col = points[:, j]
        if mx:
            r[j], z[j] = col.min(), col.max()
        else:
            r[j], z[j] = col.max(), col.min()
    return r, z


def hv_ratio(points, maximize, r, z):
    """Returns (hv_ratio, num_nondominated) for one method's archive
    `points` (raw objective space), given shared reference r / ideal z
    (raw objective space, same convention as `points`).
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 0.0, 0
    pts_max = _to_max_space(points, maximize)
    r_max = _to_max_space(r.reshape(1, -1), maximize)[0]
    z_max = _to_max_space(z.reshape(1, -1), maximize)[0]
    mask = pareto_mask_max(pts_max)
    front = pts_max[mask]
    hv = hv_2d_max(front, r_max)
    denom = abs(r_max[0] - z_max[0]) * abs(r_max[1] - z_max[1])
    ratio = hv / denom if denom > 1e-12 else 0.0
    return float(ratio), int(mask.sum())
