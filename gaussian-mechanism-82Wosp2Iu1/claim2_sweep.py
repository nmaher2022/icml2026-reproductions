# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Claim 2 eps-sweep: Haar symmetrization delta' <= delta across privacy levels."""
import sys; sys.path.insert(0, ".")
import numpy as np, csv
from claim23_family import hockey_stick_mc
from sgg_lib import delta_spherical

T, s = 2, 1.0
lam = 1.1
logpdf = lambda x: -np.abs(x).sum(1) / lam
sampler = lambda n, r: r.laplace(0, lam, (n, T))

xs = sampler(4_000_000, np.random.default_rng(3))
rad = np.sort(np.linalg.norm(xs, axis=1))
r_of_q = lambda q: np.quantile(rad, q)
hist, edges = np.histogram(rad, bins=3000, density=True)
centers = (edges[:-1] + edges[1:]) / 2
logf = np.log(np.maximum(hist, 1e-300)) - (T - 1) * np.log(np.maximum(centers, 1e-300))
logshape = lambda r: np.interp(r, centers, logf, left=logf[0], right=-np.inf)

rows = []
for eps in [0.25, 0.5, 1.0, 2.0, 3.0]:
    ds = []
    for a in np.linspace(0, np.pi / 2, 7):
        v = s * np.array([np.cos(a), np.sin(a)])
        d, se = hockey_stick_mc(logpdf, sampler, v, eps)
        ds.append(d)
    d_orig = max(ds)
    d_symm = delta_spherical(eps, r_of_q, logshape, T, s=s, n_r=600, n_w=300)
    rows.append({"eps": eps, "delta_orig": d_orig, "delta_symm": d_symm,
                 "improved": d_symm <= d_orig + 3e-4})
    print(f"eps={eps:4.2f}: worst-dir delta orig={d_orig:.6f} symm={d_symm:.6f} "
          f"improved={rows[-1]['improved']}")
with open("claim2_sweep.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("all improved:", all(r["improved"] for r in rows))
