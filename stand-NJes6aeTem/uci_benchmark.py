# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "xgboost", "ucimlrepo", "pandas"]
# ///
"""UCI noisy-dataset benchmark (arXiv 2409.07653v2 Appendix E, Table 2/3) -- Claim 7.

Six classic UCI datasets (breast-cancer, hepatitis, soybean[small], tic-tac-toe, vote, zoo),
80/20 train/test split, comparing STAND / STAND(heir) / DecisionTree / RandomForest / XGBoost.
Categorical string features are ordinal-encoded per column; missing values ('?' etc, present in
breast-cancer/hepatitis/vote) get their own integer category rather than being imputed/dropped,
matching STAND's "everything is a categorical literal" design and avoiding a lossy dropna step
that would shrink already-small datasets like hepatitis (155 rows) and soybean (47 rows).
"""
import json
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from stand_lib import STAND

DATASETS = {
    "breast-cancer": 14,
    "hepatitis": 46,
    "soybean": 91,
    "tic-tac-toe": 101,
    "vote": 105,
    "zoo": 111,
}
N_REPS = 10


def load_dataset(uci_id):
    d = fetch_ucirepo(id=uci_id)
    X = d.data.features.copy()
    y = d.data.targets.copy()
    y_col = y.columns[0]
    y = y[y_col]

    for col in X.columns:
        X[col] = X[col].astype(str).fillna("__MISSING__")
        X[col] = pd.Categorical(X[col]).codes
    X = X.to_numpy(dtype=np.int64)

    y = pd.Categorical(y.astype(str)).codes
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) > 2:
        # STAND and this reproduction's baselines are evaluated as binary precondition
        # classifiers in the paper; collapse to "majority class vs rest" for multi-class UCI
        # sets (soybean, zoo) so the comparison is apples-to-apples across all 6 datasets.
        majority = np.bincount(y).argmax()
        y = (y == majority).astype(np.int64)
    return X, y


def make_models():
    return {
        "STAND": lambda: STAND(hierarchical_shrinkage=False),
        "STAND-hs": lambda: STAND(hierarchical_shrinkage=True, lambda_p=25.0, lambda_s=25.0,
                                   lambda_n=50.0),
        "DecisionTree": lambda: DecisionTreeClassifier(criterion="gini", random_state=0),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
        "XGBoost": lambda: xgb.XGBClassifier(eval_metric="logloss", random_state=0,
                                              verbosity=0),
    }


def main():
    models = make_models()
    results = {name: {ds: [] for ds in DATASETS} for name in models}

    for ds_name, uci_id in DATASETS.items():
        X, y = load_dataset(uci_id)
        print(f"{ds_name}: n={len(y)} features={X.shape[1]} pos_frac={y.mean():.3f}")
        for rep in range(N_REPS):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=rep,
                                                        stratify=y if len(np.unique(y)) > 1
                                                        else None)
            for name, ctor in models.items():
                model = ctor()
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                acc = (pred == y_te).mean()
                results[name][ds_name].append(float(acc))

    summary = {}
    for name in models:
        per_ds_means = {ds: float(np.mean(v)) for ds, v in results[name].items()}
        summary[name] = {
            "per_dataset_mean_accuracy": per_ds_means,
            "overall_mean_accuracy": float(np.mean(list(per_ds_means.values()))),
        }

    with open("uci_results.json", "w") as f:
        json.dump({"n_reps": N_REPS, "summary": summary, "raw": results}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
