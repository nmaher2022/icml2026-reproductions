# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas"]
# ///
"""Claim 3 — stochastic Gluon (Algorithm 1), beta^k = 1-(k+1)^{-1/2},
t_i^k = t_i*(k+1)^{-3/4}, M_i^0 = grad_i f_{xi^0}(X^0), achieves O(1/K^{1/4})
convergence (briefing Theorem 2 / paper Theorem 4.3, mislabeled "Theorem 1"
in the claim extraction -- see PAPER_BRIEFING.md's cross-check section).

Metric (briefing, Theorem 2 statement):
    min_{k<K} sum_i (1/(12 L1_i)) * E[||grad_i f(X^k)||_(i)*]

Noise model: unbiased per-entry Gaussian noise added to the true gradient at
each layer/iteration (Assumption 2, bounded variance sigma^2). The
expectation is estimated by averaging the per-iteration TRUE gradient dual
norm (evaluated at the realized, noise-driven iterate X^k) across independent
noise seeds, THEN taking the running min over k -- matching the theorem's
"min_k E[...]" order (expectation inside, min outside), not "E[min_k...]".
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import gluon_common as gc

K_SWEEP = [50, 100, 200, 400, 800, 1600]
K_MAX = max(K_SWEEP)
N_SEEDS = 15
SIGMA_MULT = 2.0  # per-entry noise std = SIGMA_MULT * phi(X0) / sqrt(n_entries)


def build_setup():
    layers = gc.make_layers()
    cal = gc.calibrate_L0_L1(layers)
    weights = {L["name"]: 1.0 / (12.0 * cal[L["name"]]["L1"]) for L in layers}
    X0 = gc.init_X0(layers, radius=2.5, seed=0)
    sigma = {}
    for L, X in zip(layers, X0):
        _, phi0 = gc.grad_and_dual(L["kind"], X, L["a"], L["b"])
        sigma[L["name"]] = SIGMA_MULT * phi0 / np.sqrt(np.prod(L["shape"]))
    return layers, cal, weights, X0, sigma


def run_one_seed(layers, cal, weights, X0, sigma, seed, K_max):
    """Algorithm 1: momentum on noisy gradients, scheduled LMO stepsize.
    Records the TRUE (noiseless) weighted dual-norm at each visited X^k."""
    rng = np.random.default_rng(seed)
    X = [x.copy() for x in X0]
    M = None
    metric_seq = np.zeros(K_max)
    for k in range(K_max):
        beta_k = 1.0 - 1.0 / np.sqrt(k + 1)
        wsum_true = 0.0
        Gs_noisy = []
        for L, Xi in zip(layers, X):
            g_true, phi_true = gc.grad_and_dual(L["kind"], Xi, L["a"], L["b"])
            noise = rng.normal(0.0, sigma[L["name"]], size=L["shape"])
            Gs_noisy.append(g_true + noise)
            wsum_true += weights[L["name"]] * phi_true
        metric_seq[k] = wsum_true

        if M is None:
            M = [g.copy() for g in Gs_noisy]  # M^0 = grad f_{xi^0}(X^0), no blending
        else:
            M = [beta_k * m + (1.0 - beta_k) * g for m, g in zip(M, Gs_noisy)]

        for i, L in enumerate(layers):
            t_i_const = 1.0 / cal[L["name"]]["L1"]  # constant t_i, paper-consistent scale
            t = t_i_const * (k + 1) ** (-0.75)
            X[i] = gc.lmo_update(L["kind"], X[i], M[i], t)
    return metric_seq


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
    layers, cal, weights, X0, sigma = build_setup()

    print("=== layer setup ===")
    for L in layers:
        print(f"  {L['name']:5s} kind={L['kind']:8s} shape={L['shape']} n_channels={gc.n_channels(L)}")
    print("\n=== calibrated (L0_i, L1_i) ===")
    for name, d in cal.items():
        print(f"  {name:5s} L0={d['L0']:.4f} L1={d['L1']:.4f}")
    print("\n=== per-entry noise std (sigma_i, Assumption 2) ===")
    for name, s in sigma.items():
        print(f"  {name:5s} sigma={s:.4f}")

    # ---- SMOKETEST: tiny K, 1 seed ----
    print("\n=== SMOKETEST: K up to 50, 1 seed, checking for NaNs / sane magnitudes ===")
    smoke = run_one_seed(layers, cal, weights, X0, sigma, seed=0, K_max=50)
    assert np.all(np.isfinite(smoke)), "NaN/Inf in smoketest trajectory!"
    assert np.all(smoke >= 0), "negative metric in smoketest!"
    print(f"  smoketest metric[0]={smoke[0]:.4f} metric[49]={smoke[-1]:.4f} -- OK, no NaNs, sane magnitude")

    # quick timing check on a couple of seeds at K=200 before committing to the full sweep
    import time
    t0 = time.time()
    _ = run_one_seed(layers, cal, weights, X0, sigma, seed=1, K_max=200)
    dt = time.time() - t0
    est_total = dt * (K_MAX / 200) * N_SEEDS
    print(f"  timing: one seed at K=200 took {dt:.2f}s -> est. full run ({N_SEEDS} seeds, K={K_MAX}) ~ {est_total:.1f}s")

    # ---- FULL RUN: N_SEEDS independent noise trajectories up to K_MAX ----
    print(f"\n=== FULL RUN: {N_SEEDS} seeds, K_max={K_MAX} ===")
    all_seqs = np.zeros((N_SEEDS, K_MAX))
    for s in range(N_SEEDS):
        all_seqs[s] = run_one_seed(layers, cal, weights, X0, sigma, seed=s, K_max=K_MAX)
    assert np.all(np.isfinite(all_seqs))

    avg_traj = all_seqs.mean(axis=0)  # E[...] at each k, across seeds
    running_min = np.minimum.accumulate(avg_traj)  # min_{k<K} E[...]

    rows = []
    for K in K_SWEEP:
        rows.append(dict(K=K, metric_value=float(running_min[K - 1]), n_seeds=N_SEEDS))
    df = pd.DataFrame(rows)
    df.to_csv("claim3_stochastic_rate.csv", index=False)
    print(df.to_string(index=False))

    # also save per-seed raw (pre-min) metric at the sweep K's for transparency
    per_seed_rows = []
    for s in range(N_SEEDS):
        for K in K_SWEEP:
            per_seed_rows.append(dict(seed=s, K=K, metric_at_K=float(all_seqs[s, K - 1])))
    pd.DataFrame(per_seed_rows).to_csv("claim3_stochastic_rate_per_seed.csv", index=False)

    slope, intercept, r2 = fit_slope(df["K"].values, df["metric_value"].values)
    print(f"\nlog-log fit: slope={slope:.4f}  intercept={intercept:.4f}  R^2={r2:.4f}")
    print("claim predicts slope ~ -0.25 (O(1/K^{1/4})); tolerance window [-0.35, -0.15]")

    with open("claim3_fit_summary.txt", "w") as fh:
        fh.write(f"slope={slope:.6f}\nintercept={intercept:.6f}\nR2={r2:.6f}\nn_seeds={N_SEEDS}\n")
        fh.write("K,metric_value\n")
        for _, r in df.iterrows():
            fh.write(f"{int(r['K'])},{r['metric_value']:.8f}\n")

    print("\nWrote claim3_stochastic_rate.csv, claim3_stochastic_rate_per_seed.csv, claim3_fit_summary.txt")


if __name__ == "__main__":
    main()
