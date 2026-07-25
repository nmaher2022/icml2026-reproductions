"""
Claim 1 numerical audit: the paper proposes a *single* studentized
spherical-harmonic energy-distance two-sample test statistic T_p_mn that is
applicable, without modification beyond a choice of embedding map, to both

  - directional data: points natively on the sphere S^d (e.g. unit vectors),
  - compositional data: points on the simplex Delta^{d-1}, mapped onto a
    sphere via a fixed transform (sqrt-transform onto the positive
    orthant of S^{d-1}, or plain L2-normalization).

This script demonstrates the SAME OptimizedTestStatistic/SphericalTestConfig
code path (optimized_estimators.py) operating correctly on both data types:
it (i) returns a well-defined finite statistic in every case, (ii) behaves
like a valid test under H0 (T concentrated near 0, not systematically large),
and (iii) is clearly sensitive to genuine distributional differences under H1
(T large and positive) -- for both directional and compositional data, and
for both compositional embeddings (sqrt-transform, L2-normalize).

This is a narrower "does it work as advertised on both domains" check;
Claim 2 does the rigorous asymptotic-normality audit and Claim 3 does the
power comparison against baselines.
"""
import sys
import numpy as np

sys.path.insert(0, "nonparametric_compositional_two_sample_test")
from optimized_estimators import SphericalTestConfig, OptimizedTestStatistic
from dgp_dm_two_sample import DMScenario, generate_dm_two_sample
from spherical_simulations import random_points_on_sphere, sample_von_mises_fisher


def sqrt_transform(X):
    return np.sqrt(np.maximum(X, 0.0))


def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def summarize(T):
    return float(np.mean(T)), float(np.std(T, ddof=1)), float(np.min(T)), float(np.max(T))


def main():
    rows = []
    reps = 300

    # ---------------------------------------------------------------
    # 1) Directional data, native points on S^2 (d=2)
    # ---------------------------------------------------------------
    d = 2
    cfg = SphericalTestConfig(p=3, d=d)
    calc = OptimizedTestStatistic(cfg)

    T_h0 = np.array([calc.compute(random_points_on_sphere(150, d, seed=1000 + 2 * i),
                                   random_points_on_sphere(150, d, seed=1000 + 2 * i + 1))
                      for i in range(reps)])
    mu_rot = np.array([np.cos(np.radians(15)), np.sin(np.radians(15)), 0.0])
    mu0 = np.array([1.0, 0.0, 0.0])
    T_h1 = np.array([calc.compute(sample_von_mises_fisher(150, d, mu0, 12.0, seed=2000 + 2 * i),
                                   sample_von_mises_fisher(150, d, mu_rot, 12.0, seed=2000 + 2 * i + 1))
                      for i in range(reps)])
    rows.append(("Directional (native S^2), H0: unif vs unif", *summarize(T_h0)))
    rows.append(("Directional (native S^2), H1: vMF loc-shift 15deg", *summarize(T_h1)))

    # ---------------------------------------------------------------
    # 2) Compositional data, D=12 components, two embeddings
    # ---------------------------------------------------------------
    D = 12
    mu_base = np.linspace(1, D, D); mu_base = mu_base / mu_base.sum()
    v = np.zeros(D); v[0] = 1.0; v[1] = -1.0; v = v - v.mean()
    mu_shift = mu_base * np.exp(0.5 * v); mu_shift = mu_shift / mu_shift.sum()

    sc_h0 = DMScenario(d=D, m=100, n=100, N=150, mu0=mu_base, mu1=mu_base, kappa_base=50.0, eta=1.0)
    sc_h1 = DMScenario(d=D, m=100, n=100, N=150, mu0=mu_base, mu1=mu_shift, kappa_base=50.0, eta=1.0)

    for embed_name, embed in [("sqrt-transform", sqrt_transform), ("L2-normalize", l2_normalize)]:
        cfg_c = SphericalTestConfig(p=4, d=D - 1)
        calc_c = OptimizedTestStatistic(cfg_c)

        T_h0c = np.empty(reps)
        T_h1c = np.empty(reps)
        for i in range(reps):
            out0 = generate_dm_two_sample(sc_h0, seed=3000 + i)
            T_h0c[i] = calc_c.compute(embed(out0["comps0"]), embed(out0["comps1"]))
            out1 = generate_dm_two_sample(sc_h1, seed=4000 + i)
            T_h1c[i] = calc_c.compute(embed(out1["comps0"]), embed(out1["comps1"]))

        rows.append((f"Compositional (D={D}, {embed_name}), H0: same DM mean", *summarize(T_h0c)))
        rows.append((f"Compositional (D={D}, {embed_name}), H1: sparse mean-shift(delta=0.5)", *summarize(T_h1c)))

    print("\n### T_p_mn behavior across data types / embeddings (same code path, n=%d reps each)\n" % reps)
    print("| Setting | mean(T) | std(T) | min(T) | max(T) |")
    print("|---|---:|---:|---:|---:|")
    for name, m, s, lo, hi in rows:
        print(f"| {name} | {m:.4f} | {s:.4f} | {lo:.4f} | {hi:.4f} |")

    print("\nExpectation: under H0, mean(T)~0 and std(T)~1 for every data type/embedding "
          "(same OptimizedTestStatistic code path). Under H1, mean(T) is large and positive "
          "(the test is one-sided: large T => reject H0), confirming the single studentized "
          "spherical-harmonic energy-distance statistic is directly applicable to both "
          "directional and compositional data.")


if __name__ == "__main__":
    main()
