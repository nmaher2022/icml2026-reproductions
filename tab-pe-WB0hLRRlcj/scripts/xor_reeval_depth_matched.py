"""Supplementary re-evaluation for the XOR stress test (Claim 4), NOT part of the official
DPSDA/Tab-PE codebase.

Why this script exists: the official xor_stress_test.py evaluates downstream utility with
model_name="tabpfn" (TabPFN, a transformer that can capture arbitrary-order feature interactions
without hyperparameter tuning). Our reproduction had to substitute model_name="xgboost" for
xor_stress_test_xgb.py because of TabPFN's interactive license gate (see BUGFIX_LOG.md). But the
paper's own Appendix C.1 (paper.txt lines 1061-1063) states explicitly: "The max depth of XGBoost
must be equal to the number of features to achieve better-than-random accuracy" on XOR data.
`pe/callback/tabular/classifier.py`'s TabClassifier hardcodes `xgb.XGBClassifier(objective=
"binary:logistic")` with no max_depth override, i.e. XGBoost's library default (max_depth=6).
That is why xor_stress_test_xgb.py's own default-depth XGBoost eval collapsed to ~50% AUC at
4-5 features (see REPRO_LOG.md) even though the *synthetic data itself* may well be correct --
this script checks that, by re-running just the classifier evaluation (not the DP generation)
with max_depth explicitly set equal to num_features, on the exact same already-generated
synthetic data (loaded from each run's saved checkpoint) and the exact same test set.

This does NOT touch or re-run the DP synthetic-data generation (that already completed and is
disclosed/logged as-is) -- it only asks "was our classifier substitution itself producing a
misleading null result," per the paper's own documented XGBoost-depth caveat.
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from pe.data import Data, TabularCSV
from pe.constant.data import (
    TABULAR_DATA_COLUMN_NAME,
    LABEL_ID_COLUMN_NAME,
    VARIATION_API_FOLD_ID_COLUMN_NAME,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def evaluate(num_features):
    exp_folder = os.path.join(RESULTS_DIR, f"xor_stress_test_{num_features}_features")
    checkpoint_path = os.path.join(exp_folder, "checkpoint")

    syn_data = Data()
    syn_data.load_checkpoint(checkpoint_path)
    syn_data = syn_data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})

    test_data = TabularCSV(
        csv_path="https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        f"heads/main/tabular/sim/xor-stress-test/{num_features}_feature_xor/data_test.csv",
        metadata_path="https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        f"heads/main/tabular/sim/xor-stress-test/{num_features}_feature_xor/metadata.json",
    )

    feature_columns = test_data.metadata["feature_columns"]
    cat_columns = syn_data.metadata["cat_columns"]

    syn_df = pd.DataFrame(syn_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=feature_columns)
    test_df = pd.DataFrame(test_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=feature_columns)
    syn_df[LABEL_ID_COLUMN_NAME] = syn_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()
    test_df[LABEL_ID_COLUMN_NAME] = test_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()

    for column in feature_columns + [LABEL_ID_COLUMN_NAME]:
        merged = pd.concat([syn_df[column], test_df[column]])
        if column in cat_columns + [LABEL_ID_COLUMN_NAME]:
            enc = LabelEncoder()
            enc.fit(merged.values)
            syn_df[column] = enc.transform(syn_df[column].values)
            test_df[column] = enc.transform(test_df[column].values)
        else:
            scaler = MinMaxScaler()
            scaler.fit(merged.values.reshape(-1, 1))
            syn_df[column] = scaler.transform(syn_df[column].values.reshape(-1, 1))
            test_df[column] = scaler.transform(test_df[column].values.reshape(-1, 1))

    X_train = syn_df.drop(LABEL_ID_COLUMN_NAME, axis=1).values
    y_train = syn_df[LABEL_ID_COLUMN_NAME].values
    X_test = test_df.drop(LABEL_ID_COLUMN_NAME, axis=1).values
    y_test = test_df[LABEL_ID_COLUMN_NAME].values

    results = {"num_features": num_features, "num_train": len(X_train), "num_test": len(X_test)}
    for depth in sorted(set([num_features, 6])):
        clf = xgb.XGBClassifier(objective="binary:logistic", max_depth=depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred, average="macro") * 100
        auc = roc_auc_score(y_test, y_proba) * 100
        results[f"max_depth_{depth}"] = {"accuracy": acc, "macro_f1": f1, "auc": auc}
        print(f"num_features={num_features} max_depth={depth}: acc={acc:.2f}% f1={f1:.2f} auc={auc:.2f}")

    return results


if __name__ == "__main__":
    all_results = {}
    for n in [1, 2, 3, 4, 5]:
        all_results[n] = evaluate(n)

    out_path = os.path.join(RESULTS_DIR, "xor_depth_matched_eval.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")
