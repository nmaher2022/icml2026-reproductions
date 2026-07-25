"""
Claim 2 numerical audit: Under H0, the studentized test statistic T_p_mn
converges in distribution to N(0,1) -- WITHOUT any permutation or bootstrap
calibration. We check this directly:

  (a) For several (data-type, m, n, p, d) configurations under H0, draw many
      Monte Carlo replicates of T_p_mn and compare its empirical distribution
      to N(0,1): mean, std, skewness, kurtosis, a KS test against N(0,1), and
      the empirical rejection rate at nominal alpha in {0.01, 0.05, 0.10}
      using ASYMPTOTIC N(0,1) critical values only (no resampling anywhere).
      If the claim holds, rejection rate ~ nominal alpha, and this should
      hold already at moderate N and improve (or stay good) as N grows.

  (b) A control that relaxes the paper's bias-correction: the same statistic
      computed with the NAIVE (non bias-corrected) variance estimator
      V_p_mn instead of the paper's bias-corrected V_p_mn_unbiased. If the
      bias correction is load-bearing for the "known-scale CLT, no
      permutation needed" claim, the naive version should show a systematic
      departure from N(0,1) (mis-sized test).

Both directional (points on S^d, uniform/vMF) and compositional (Dirichlet-
Multinomial compositions mapped to the sphere via sqrt-transform) data types
are exercised, since Claim 3 explicitly says the test targets both.

Output is a small set of aggregate tables -- no per-trial dumps. Progress is
printed per-configuration (not per-replicate) to keep output compact.
"""
import sys
import time
import numpy as np
from scipy import stats

sys.path.insert(0, "nonparametric_compositional_two_sample_test")
from optimized_estimators import SphericalTestConfig, OptimizedTestStatistic
from dgp_dm_two_sample import DMScenario, generate_dm_two_sample
from spherical_simulations import random_points_on_sphere, sample_von_mises_fisher


def sqrt_transform(X):
    return np.sqrt(np.maximum(X, 0.0))


def run_null_config(name, sampler_X, sampler_Y, p, d, num_reps, use_unbiased=True):
    cfg = SphericalTestConfig(p=p, d=d)
    calc = OptimizedTestStatistic(cfg)
    T = np.empty(num_reps)
    for i in range(num_reps):
        X = sampler_X(seed=10_000 + 2 * i)
        Y = sampler_Y(seed=10_000 + 2 * i + 1)
        T[i] = calc.compute(X, Y, use_unbiased=use_unbiased)

    mean, std = float(np.mean(T)), float(np.std(T, ddof=1))
    skew, kurt = float(stats.skew(T)), float(stats.kurtosis(T))
    ks_stat, ks_p = stats.kstest(T, stats.norm.cdf)

    alphas = [0.01, 0.05, 0.10]
    emp_size = {a: float(np.mean(T > stats.norm.isf(a))) for a in alphas}

    print(f"  done: {name} (p={p}, d={d}, reps={num_reps})")
    return {
        "name": name, "p": p, "d": d, "N": num_reps,
        "mean": mean, "std": std, "skew": skew, "kurt": kurt,
        "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        "size@1%": emp_size[0.01], "size@5%": emp_size[0.05], "size@10%": emp_size[0.10],
    }


def fmt_row(r):
    return (f"| {r['name']:<26} | {r['d']:>3} | {r['p']:>2} | {r['N']:>5} | "
            f"{r['mean']:>7.4f} | {r['std']:>6.4f} | {r['skew']:>7.4f} | {r['kurt']:>7.4f} | "
            f"{r['ks_stat']:>6.4f} | {r['ks_p']:>7.4f} | "
            f"{r['size@1%']:>6.3f} | {r['size@5%']:>6.3f} | {r['size@10%']:>6.3f} |")


HEADER = ("| Config                     |   d |  p |     N |    mean |    std |    skew |    kurt | "
          "KS stat |   KS p | size@1% | size@5% | size@10% |")
SEP =    ("|----------------------------|-----|----|-------|---------|--------|---------|---------|"
          "--------|--------|---------|---------|----------|")


def main():
    t_start = time.time()
    print("Running Claim 2 audit (this prints one line per finished configuration)...")
    results = []

    # ---- (a) Bias-corrected statistic under H0, directional data ----
    d = 2
    for m, n, p, reps in [(40, 40, 2, 2000), (100, 100, 3, 1200), (300, 300, 3, 400)]:
        sampler_X = lambda seed, m=m, d=d: random_points_on_sphere(m, d, seed=seed)
        sampler_Y = lambda seed, n=n, d=d: random_points_on_sphere(n, d, seed=seed)
        r = run_null_config(f"Sph-Uniform m=n={m} (S^{d})", sampler_X, sampler_Y, p, d, reps)
        results.append(r)

    # vMF-vMF, same params (H0), moderate concentration, d=9
    d = 9
    mu = np.zeros(d + 1); mu[0] = 1.0
    kappa = 15.0
    for m, n, p, reps in [(120, 120, 4, 800), (200, 200, 4, 400)]:
        sampler_X = lambda seed, m=m: sample_von_mises_fisher(m, d, mu, kappa, seed=seed)
        sampler_Y = lambda seed, n=n: sample_von_mises_fisher(n, d, mu, kappa, seed=seed)
        r = run_null_config(f"Sph-vMF(k={kappa:.0f}) m=n={m} (S^{d})", sampler_X, sampler_Y, p, d, reps)
        results.append(r)

    # ---- (a) Bias-corrected statistic under H0, compositional data ----
    d_comp = 10
    mu0 = np.linspace(1, d_comp, d_comp); mu0 = mu0 / mu0.sum()
    for N_lib, label, reps in [(200, "dense(~5%z)", 500), (20, "sparse(~50%z)", 500)]:
        m = n = 100
        sc = DMScenario(d=d_comp, m=m, n=n, N=N_lib, mu0=mu0, mu1=mu0, kappa_base=50.0, eta=1.0)

        def sampler_X(seed, sc=sc):
            out = generate_dm_two_sample(sc, seed=seed)
            return sqrt_transform(out["comps0"])

        def sampler_Y(seed, sc=sc):
            out = generate_dm_two_sample(sc, seed=seed)
            return sqrt_transform(out["comps1"])

        p = 4
        r = run_null_config(f"Comp-DM {label} m=n={m}", sampler_X, sampler_Y, p, d_comp - 1, reps)
        results.append(r)

    print("\n### (a) Bias-corrected T_p_mn under H0 vs N(0,1) -- both data types\n")
    print(HEADER)
    print(SEP)
    for r in results:
        print(fmt_row(r))

    # ---- (b) CONTROL: relax the bias-correction assumption ----
    # Bias term a_0_p^2/((N-1)(N-3)) grows fast with p,d and shrinks with N,
    # so we pick configs spanning "bias term negligible" -> "bias term huge
    # relative to V_p_mn" to show the naive (non-corrected) statistic
    # degrades exactly where theory predicts it should.
    control_configs = [
        ("Sph-Uniform m=n=100 (S^2)", 2, 3, 100, 1200),   # bias term tiny -> expect ~same as (a)
        ("Sph-Uniform m=n=20 (S^2)", 2, 7, 20, 1500),      # bias term ~2.0, moderate N -> some effect
        ("Sph-Uniform m=n=30 (S^9)", 9, 4, 30, 1500),      # bias term >> V_p_mn -> should badly miscalibrate
    ]
    control_results = []
    for name, d, p, m, reps in control_configs:
        sampler_X = lambda seed, m=m, d=d: random_points_on_sphere(m, d, seed=seed)
        sampler_Y = lambda seed, n=m, d=d: random_points_on_sphere(n, d, seed=seed)
        r = run_null_config(f"{name} [NAIVE V]", sampler_X, sampler_Y, p, d, reps,
                             use_unbiased=False)
        control_results.append(r)

    print("\n### (b) CONTROL -- same configs under H0, bias-correction *removed* (naive V_p_mn)\n")
    print(HEADER)
    print(SEP)
    for r in control_results:
        print(fmt_row(r))

    print(f"\nTotal wall-clock: {time.time() - t_start:.1f}s. Nominal size targets: 1%, 5%, 10%.")
    print("Reference for (a): mean~0, std~1, KS p-value not small (ideally >0.05), "
          "empirical size close to nominal alpha, all WITHOUT any permutation/bootstrap step.")
    print("Reference for (b): if bias-correction matters, mean/size should be visibly "
          "off from (a)'s bias-corrected results at the same (m,n,p).")


if __name__ == "__main__":
    main()
