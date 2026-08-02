# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "xgboost"]
# ///
"""Synthetic precondition-induction benchmark driver (arXiv 2409.07653v2 Sec 6, Appendix B).

Reproduces claims 1-6 from PAPER_BRIEFING.md using data_gen.generate_synthetic_dataset and
stand_lib.STAND, compared against sklearn Decision Tree / Random Forest / MLP and XGBoost.

TOY-SCALE REDUCTIONS from the paper (documented in VERDICTS.md):
- N_REPS=15 repetitions instead of the paper's 100 (still enough for a meaningful mean +/-
  standard error given the paper's own effect sizes are large, e.g. 90% vs 56% accuracy).
- Incremental training curve sampled at N=10,20,...,100 (10 checkpoints) instead of every
  single example N=1..100, refitting each model from scratch at each checkpoint (STAND has no
  incremental/warm-start fit implemented).
- Claim 5 (active learning utility) uses a *batch* active-learning approximation: instead of
  picking one most-uncertain example at a time and refitting after each pick (combinatorially
  expensive at this scale), the current model's uncertainty ranking over the remaining pool is
  computed once per checkpoint and the top 10 most-uncertain pool examples are added together.
"""
import json
import time
import warnings

import numpy as np
from data_gen import generate_synthetic_dataset
from stand_lib import STAND
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_REPS = 15
CHECKPOINTS = list(range(10, 101, 10))
POOL_SIZE = 200
ACTIVE_BATCH = 10


def make_models():
    return {
        "STAND": lambda: STAND(hierarchical_shrinkage=False),
        "STAND-hs": lambda: STAND(hierarchical_shrinkage=True, lambda_p=25.0, lambda_s=25.0,
                                   lambda_n=50.0),
        "DecisionTree": lambda: DecisionTreeClassifier(criterion="gini", random_state=0),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
        "XGBoost": lambda: xgb.XGBClassifier(eval_metric="logloss", random_state=0,
                                              verbosity=0),
        "NeuralNet": lambda: MLPClassifier(hidden_layer_sizes=(100, 100, 100),
                                            activation="relu", solver="adam",
                                            learning_rate_init=1e-3, max_iter=300,
                                            random_state=0),
    }


def safe_fit_predict_proba(ctor, X_tr, y_tr, X_eval):
    """Fit a fresh model instance; fall back to a constant majority-class predictor if the
    training slice has only one class present (sklearn/xgboost require >=2 classes to fit)."""
    classes = np.unique(y_tr)
    if len(classes) < 2:
        const = float(classes[0])
        return np.full(len(X_eval), const)
    model = ctor()
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba_pos"):
        return model.predict_proba_pos(X_eval)
    return model.predict_proba(X_eval)[:, 1]


def run_rep(seed, models):
    d = generate_synthetic_dataset(seed=seed, pool_size=POOL_SIZE)
    X_train, y_train = d["X_train"], d["y_train"]
    X_hold, y_hold = d["X_holdout"], d["y_holdout"]
    X_pool, y_pool = d["X_pool"], d["y_pool"]

    regular = {name: {} for name in models}
    for N in CHECKPOINTS:
        for name, ctor in models.items():
            proba = safe_fit_predict_proba(ctor, X_train[:N], y_train[:N], X_hold)
            regular[name][N] = proba

    active = {name: {} for name in models}
    for name, ctor in models.items():
        act_X = list(X_train[:CHECKPOINTS[0]])
        act_y = list(y_train[:CHECKPOINTS[0]])
        pool_X = list(X_pool)
        pool_y = list(y_pool)
        proba = safe_fit_predict_proba(ctor, np.array(act_X), np.array(act_y), X_hold)
        active[name][CHECKPOINTS[0]] = proba
        for N in CHECKPOINTS[1:]:
            classes = np.unique(act_y)
            if len(classes) < 2 or not pool_X:
                unc = np.zeros(len(pool_X))
            else:
                model = ctor()
                model.fit(np.array(act_X), np.array(act_y))
                if hasattr(model, "predict_proba_pos"):
                    pool_proba = model.predict_proba_pos(np.array(pool_X))
                else:
                    pool_proba = model.predict_proba(np.array(pool_X))[:, 1]
                unc = -np.abs(pool_proba - 0.5)  # more uncertain = larger (less negative)
            k = min(ACTIVE_BATCH, len(pool_X))
            pick = np.argsort(-unc)[:k] if k > 0 else np.array([], dtype=int)
            pick = sorted(pick, reverse=True)
            for idx in pick:
                act_X.append(pool_X.pop(idx))
                act_y.append(pool_y.pop(idx))
            proba = safe_fit_predict_proba(ctor, np.array(act_X), np.array(act_y), X_hold)
            active[name][N] = proba

    return {
        "y_hold": y_hold,
        "regular": regular,
        "active": active,
    }


def main():
    models = make_models()
    t0 = time.time()
    reps = []
    for seed in range(N_REPS):
        t_rep0 = time.time()
        reps.append(run_rep(seed, models))
        print(f"rep {seed+1}/{N_REPS} done in {time.time()-t_rep0:.1f}s "
              f"(elapsed {time.time()-t0:.1f}s)", flush=True)

    # ---- Claim 1: holdout accuracy at N=100 ----
    acc_at_100 = {name: [] for name in models}
    for rep in reps:
        y_hold = rep["y_hold"]
        for name in models:
            proba = rep["regular"][name][100]
            pred = (proba >= 0.5).astype(int)
            acc_at_100[name].append((pred == y_hold).mean())

    # ---- Claim 2: FP/FN reoccurrence (examples wrong at >=2 distinct checkpoints) ----
    reocc = {name: {"fp": [], "fn": []} for name in models}
    for rep in reps:
        y_hold = rep["y_hold"]
        neg_mask = y_hold == 0
        pos_mask = y_hold == 1
        for name in models:
            wrong_counts = np.zeros(len(y_hold))
            for N in CHECKPOINTS:
                proba = rep["regular"][name][N]
                pred = (proba >= 0.5).astype(int)
                wrong_counts += (pred != y_hold).astype(int)
            fp_rate = (wrong_counts[neg_mask] >= 2).mean() if neg_mask.any() else 0.0
            fn_rate = (wrong_counts[pos_mask] >= 2).mean() if pos_mask.any() else 0.0
            reocc[name]["fp"].append(fp_rate)
            reocc[name]["fn"].append(fn_rate)

    # ---- Claim 3: productive monotonicity (>2% confidence-toward-truth changes) ----
    prod_mono = {name: [] for name in models}
    for rep in reps:
        y_hold = rep["y_hold"]
        for name in models:
            productive = 0
            unproductive = 0
            probas = [rep["regular"][name][N] for N in CHECKPOINTS]
            for i in range(1, len(probas)):
                conf_prev = np.where(y_hold == 1, probas[i - 1], 1 - probas[i - 1])
                conf_now = np.where(y_hold == 1, probas[i], 1 - probas[i])
                delta = conf_now - conf_prev
                mask = np.abs(delta) > 0.02
                productive += int((delta[mask] > 0).sum())
                unproductive += int((delta[mask] < 0).sum())
            total = productive + unproductive
            prod_mono[name].append(productive / total if total > 0 else np.nan)

    # ---- Claim 4: calibration -- precision at predicted-probability ~= 100% bin ----
    calib_100 = {name: [] for name in models}
    for rep in reps:
        y_hold = rep["y_hold"]
        for name in models:
            proba = rep["regular"][name][100]
            top_mask = proba >= 0.99
            if top_mask.any():
                calib_100[name].append((y_hold[top_mask] == 1).mean())

    # ---- Claim 5: active learning utility ----
    active_util = {name: [] for name in models}
    for rep in reps:
        y_hold = rep["y_hold"]
        for name in models:
            reg_errs, act_errs = [], []
            for N in CHECKPOINTS:
                reg_pred = (rep["regular"][name][N] >= 0.5).astype(int)
                act_pred = (rep["active"][name][N] >= 0.5).astype(int)
                reg_errs.append(1 - (reg_pred == y_hold).mean())
                act_errs.append(1 - (act_pred == y_hold).mean())
            avg_reg_err = np.mean(reg_errs)
            act_acc = 1 - np.mean(act_errs)
            active_util[name].append(act_acc / avg_reg_err if avg_reg_err > 1e-9 else np.nan)

    results = {
        "n_reps": N_REPS,
        "checkpoints": CHECKPOINTS,
        "claim1_accuracy_at_100": {n: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                        "vals": [float(x) for x in v]}
                                    for n, v in acc_at_100.items()},
        "claim2_reoccurrence": {n: {"fp_mean": float(np.mean(r["fp"])),
                                     "fn_mean": float(np.mean(r["fn"]))}
                                 for n, r in reocc.items()},
        "claim3_productive_monotonicity": {n: {"mean": float(np.nanmean(v)),
                                                 "std": float(np.nanstd(v))}
                                            for n, v in prod_mono.items()},
        "claim4_calibration_at_100pct": {n: {"mean": float(np.mean(v)) if v else None,
                                              "n_examples": len(v)}
                                          for n, v in calib_100.items()},
        "claim5_active_learning_utility": {n: {"mean": float(np.nanmean(v)),
                                                "std": float(np.nanstd(v))}
                                            for n, v in active_util.items()},
    }

    with open("synthetic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
