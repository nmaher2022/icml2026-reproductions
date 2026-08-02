# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Hyperparameter sensitivity of lambda_p (arXiv 2409.07653v2 Appendix D, Fig. 12) -- Claim 6.

Paper sweeps lambda_p over {0, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0} (lambda_s, lambda_n held at their
chosen values 25.0/50.0), reporting effects on final holdout accuracy, productive monotonicity,
and error reoccurrence. Reduced here to N_REPS repetitions (toy-scale, see VERDICTS.md) at the
final N=100 checkpoint only (no incremental curve needed for this claim's headline statement).
"""
import json
import numpy as np
from data_gen import generate_synthetic_dataset
from stand_lib import STAND

LAMBDA_P_VALUES = [0, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
N_REPS = 12


def main():
    results = {lp: {"acc": [], "fp_reocc": [], "fn_reocc": []} for lp in LAMBDA_P_VALUES}
    checkpoints = [50, 100]  # two points is enough to detect reocc. (repeat wrong at both)

    for seed in range(N_REPS):
        d = generate_synthetic_dataset(seed=seed)
        X_train, y_train = d["X_train"], d["y_train"]
        X_hold, y_hold = d["X_holdout"], d["y_holdout"]
        neg_mask = y_hold == 0
        pos_mask = y_hold == 1

        for lp in LAMBDA_P_VALUES:
            wrong_counts = np.zeros(len(y_hold))
            final_pred = None
            for N in checkpoints:
                m = STAND(hierarchical_shrinkage=True, lambda_p=float(lp), lambda_s=25.0,
                           lambda_n=50.0)
                m.fit(X_train[:N], y_train[:N])
                pred = m.predict(X_hold)
                wrong_counts += (pred != y_hold).astype(int)
                final_pred = pred
            acc = (final_pred == y_hold).mean()
            fp = (wrong_counts[neg_mask] >= 2).mean() if neg_mask.any() else 0.0
            fn = (wrong_counts[pos_mask] >= 2).mean() if pos_mask.any() else 0.0
            results[lp]["acc"].append(float(acc))
            results[lp]["fp_reocc"].append(float(fp))
            results[lp]["fn_reocc"].append(float(fn))
        print(f"rep {seed+1}/{N_REPS} done", flush=True)

    summary = {
        str(lp): {
            "acc_mean": float(np.mean(v["acc"])),
            "acc_std": float(np.std(v["acc"])),
            "fp_reocc_mean": float(np.mean(v["fp_reocc"])),
            "fn_reocc_mean": float(np.mean(v["fn_reocc"])),
        }
        for lp, v in results.items()
    }
    with open("lambda_p_sweep_results.json", "w") as f:
        json.dump({"n_reps": N_REPS, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
