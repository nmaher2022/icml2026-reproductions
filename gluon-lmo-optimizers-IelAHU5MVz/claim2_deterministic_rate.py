# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas"]
# ///
"""Claim 2 — deterministic Gluon (Algorithm 2) with the adaptive per-layer
stepsize t_i^k = ||grad_i f(X^k)||_(i)* / (L0_i + L1_i ||grad_i f(X^k)||_(i)*)
achieves O(1/K^{1/2}) convergence (briefing Theorem 1 / paper Theorem 4.1).

Metric (briefing, Theorem 1 statement):
    min_{k<K} sum_i [(1/L1_i) / mean_j(1/L1_j)] * ||grad_i f(X^k)||_(i)*

We run ONE long deterministic trajectory (K_max=1600) on the synthetic
multi-timescale layered objective in gluon_common.py, record the weighted
per-iteration dual-norm sum at every step, and for each K in the sweep take
the running minimum over the first K iterates (valid since a K-step prefix
of a longer run is itself a valid K-step run of the same deterministic
algorithm). Fit log(metric) vs log(K) by least squares; the claim predicts
slope ~ -1/2 (tolerance -0.6 to -0.4 per the reproduction protocol).
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import gluon_common as gc

K_SWEEP = [50, 100, 200, 400, 800, 1600]
K_MAX = max(K_SWEEP)


def run_trajectory(layers, cal, X0, K_max, weights):
    X = [x.copy() for x in X0]
    metric_seq = np.zeros(K_max)
    for k in range(K_max):
        wsum = 0.0
        for i, L in enumerate(layers):
            G, phi = gc.grad_and_dual(L["kind"], X[i], L["a"], L["b"])
            L0, L1 = cal[L["name"]]["L0"], cal[L["name"]]["L1"]
            t = phi / (L0 + L1 * phi) if phi > 0 else 0.0
            X[i] = gc.lmo_update(L["kind"], X[i], G, t)
            wsum += weights[L["name"]] * phi
        metric_seq[k] = wsum
    return metric_seq


def build_setup():
    layers = gc.make_layers()
    cal = gc.calibrate_L0_L1(layers)
    inv_L1 = {L["name"]: 1.0 / cal[L["name"]]["L1"] for L in layers}
    mean_inv_L1 = float(np.mean(list(inv_L1.values())))
    weights = {name: v / mean_inv_L1 for name, v in inv_L1.items()}
    X0 = gc.init_X0(layers, radius=2.5, seed=0)
    return layers, cal, weights, X0


def fit_slope(Ks, vals):
    logK = np.log(Ks)
    logv = np.log(vals)
    slope, intercept = np.polyfit(logK, logv, 1)
    resid = logv - (slope * logK + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((logv - logv.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def main():
    layers, cal, weights, X0 = build_setup()

    print("=== layer setup ===")
    for L in layers:
        print(f"  {L['name']:5s} kind={L['kind']:8s} shape={L['shape']} "
              f"n_channels={gc.n_channels(L)} a=[{L['a'].min():.2e},{L['a'].max():.2e}] b={L['b']}")

    print("\n=== calibrated (L0_i, L1_i) [empirical upper certificate for Assumption 1] ===")
    for name, d in cal.items():
        print(f"  {name:5s} L0={d['L0']:.4f} L1={d['L1']:.4f} (n_pairs={d['n_pairs']})")

    val = gc.validate_assumption1(layers, cal)
    print("\n=== held-out validation of Assumption 1 (fresh sample, realistic single-step pairs) ===")
    all_pass = True
    for name, d in val.items():
        print(f"  {name:5s} pass_rate={d['pass_rate']:.4f} max_LHS/bound_ratio={d['max_ratio']:.3f} n={d['n_tot']}")
        if d["pass_rate"] < 0.98:
            all_pass = False
    print(f"  -> assumption 1 empirically holds cleanly: {all_pass}")

    Delta0 = sum(gc.f_value(L["kind"], X, L["a"], L["b"]) for L, X in zip(layers, X0))
    print(f"\nDelta0 = f(X0) - inf f = {Delta0:.4f}  (inf f = 0, all-zero minimizer)")

    # ---- SMOKETEST: tiny K sweep first ----
    print("\n=== SMOKETEST: K up to 50, checking for NaNs / sane magnitudes ===")
    smoke_metric = run_trajectory(layers, cal, X0, 50, weights)
    assert np.all(np.isfinite(smoke_metric)), "NaN/Inf in smoketest trajectory!"
    assert np.all(smoke_metric >= 0), "negative metric in smoketest!"
    assert smoke_metric[0] < smoke_metric[-1] * 1e6, "metric exploded in smoketest!"
    print(f"  smoketest metric[0]={smoke_metric[0]:.4f} metric[49]={smoke_metric[-1]:.4f} -- OK, no NaNs, sane magnitude")

    # ---- FULL RUN ----
    print(f"\n=== FULL RUN: K_max={K_MAX} ===")
    metric_seq = run_trajectory(layers, cal, X0, K_MAX, weights)
    assert np.all(np.isfinite(metric_seq))
    running_min = np.minimum.accumulate(metric_seq)

    rows = []
    for K in K_SWEEP:
        rows.append(dict(K=K, metric_value=float(running_min[K - 1])))
    df = pd.DataFrame(rows)
    df.to_csv("claim2_deterministic_rate.csv", index=False)
    print(df.to_string(index=False))

    slope, intercept, r2 = fit_slope(df["K"].values, df["metric_value"].values)
    print(f"\nlog-log fit: slope={slope:.4f}  intercept={intercept:.4f}  R^2={r2:.4f}")
    print("claim predicts slope ~ -0.5 (O(1/K^{1/2})); tolerance window [-0.6, -0.4]")

    with open("claim2_fit_summary.txt", "w") as fh:
        fh.write(f"slope={slope:.6f}\nintercept={intercept:.6f}\nR2={r2:.6f}\n")
        fh.write(f"Delta0={Delta0:.6f}\n")
        fh.write("K,metric_value\n")
        for _, r in df.iterrows():
            fh.write(f"{int(r['K'])},{r['metric_value']:.8f}\n")

    print("\nWrote claim2_deterministic_rate.csv and claim2_fit_summary.txt")


if __name__ == "__main__":
    main()
