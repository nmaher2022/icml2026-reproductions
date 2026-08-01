"""Best-effort AIM baseline for a same-hardware comparison against our own Tab-PE run.

Not part of the official DPSDA/Tab-PE codebase. Uses the AIM mechanism from its original
author's own `private-pgm` repo (github.com/ryan112358/private-pgm, cloned read-only to
DPSDA_upstream_aim/), via the `mbi` PyPI package. Degree-2 (all pairwise marginals) workload,
a single reasonable choice rather than the paper's degree-2-to-5 sweep with best-result
reporting (App. C.3) -- disclosed in REPRO_LOG.md / BUGFIX_LOG.md / VERDICTS.md, not silently
presented as matching the paper's tuned AIM baseline exactly.

Numerical columns are discretized into quantile bins (binning choice not specified by the
paper's own AIM baseline, which the paper says uses "PrivTree" -- also disclosed).
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

AIM_MECH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DPSDA_upstream_aim", "mechanisms")
sys.path.insert(0, AIM_MECH_DIR)

from mbi import Dataset, Domain  # noqa: E402
from aim import AIM  # noqa: E402


def discretize(df, int_columns, float_columns, label_column, num_bins=20):
    """Map every column to nonneg integer codes. Returns (discretized_df, decoders)."""
    decoders = {}
    out = pd.DataFrame(index=df.index)

    for col in int_columns:
        lo = df[col].min()
        out[col] = (df[col] - lo).astype(int)
        decoders[col] = ("int", lo)

    for col in float_columns:
        binned, edges = pd.qcut(df[col], q=num_bins, retbins=True, duplicates="drop")
        codes = binned.cat.codes.astype(int)
        out[col] = codes
        midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        decoders[col] = ("float", midpoints)

    codes, uniques = pd.factorize(df[label_column], sort=True)
    out[label_column] = codes
    decoders[label_column] = ("cat", list(uniques))

    return out, decoders


def undiscretize(df, decoders):
    out = pd.DataFrame(index=df.index)
    for col, (kind, info) in decoders.items():
        if kind == "int":
            out[col] = df[col].astype(int) + info
        elif kind == "float":
            midpoints = info
            idx = df[col].astype(int).clip(0, len(midpoints) - 1)
            out[col] = [midpoints[i] for i in idx]
        elif kind == "cat":
            uniques = info
            idx = df[col].astype(int).clip(0, len(uniques) - 1)
            out[col] = [uniques[i] for i in idx]
    return out


def run_aim(train_disc_df, cardinalities, epsilon, delta, degree=2, max_cells=10000, num_synth_rows=None):
    domain = Domain(list(cardinalities.keys()), list(cardinalities.values()))
    data = Dataset(train_disc_df, domain)

    workload = list(itertools.combinations(data.domain, degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= max_cells]
    workload = [(cl, 1.0) for cl in workload]

    mech = AIM(epsilon=epsilon, delta=delta)
    t0 = time.time()
    model, synth = mech.run(data, workload, num_synth_rows=num_synth_rows or len(train_disc_df))
    elapsed = time.time() - t0
    return synth.df, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["artificial_characters", "person_activity"])
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()

    base = "https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular/real/"
    name_map = {"artificial_characters": "artificial-characters", "person_activity": "person-activity"}
    ds_name = name_map[args.dataset]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"{args.dataset}_aim")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(f"{base}{ds_name}/{ds_name}_train.csv")
    test_df = pd.read_csv(f"{base}{ds_name}/{ds_name}_test.csv")
    meta = json.load(__import__("urllib.request", fromlist=["urlopen"]).urlopen(f"{base}{ds_name}/{ds_name}_metadata.json"))

    cat_columns = meta["cat_columns"]
    int_columns = meta["int_columns"]
    float_columns = meta["float_columns"]
    label_column = meta["label_columns"][0]

    print(f"Dataset {args.dataset}: {len(train_df)} train rows, cat={cat_columns}, int={int_columns}, "
          f"float={float_columns}, label={label_column}")

    # Treat any true categorical columns as already-discrete (factorize like the label).
    all_int_columns = list(int_columns)
    decoders_cat = {}
    disc_df, decoders = discretize(train_df, all_int_columns, float_columns, label_column, num_bins=args.num_bins)
    for col in cat_columns:
        codes, uniques = pd.factorize(train_df[col], sort=True)
        disc_df[col] = codes
        decoders[col] = ("cat", list(uniques))

    cardinalities = {col: int(disc_df[col].max()) + 1 for col in disc_df.columns}
    print("Cardinalities:", cardinalities)

    n = len(train_df)
    delta = 1.0 / (n * np.log(n))
    print(f"epsilon=1.0, delta={delta}, degree={args.degree}")

    synth_disc_df, elapsed = run_aim(disc_df, cardinalities, epsilon=1.0, delta=delta, degree=args.degree)
    print(f"AIM run finished in {elapsed:.1f}s, produced {len(synth_disc_df)} synthetic rows")

    synth_df = undiscretize(synth_disc_df, decoders)
    synth_df = synth_df[train_df.columns]
    synth_path = os.path.join(out_dir, "synthetic.csv")
    synth_df.to_csv(synth_path, index=False)

    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump({"dataset": args.dataset, "elapsed_seconds": elapsed, "degree": args.degree,
                    "num_bins": args.num_bins, "epsilon": 1.0, "delta": delta}, f, indent=2)

    # Evaluate with the same TabICL classifier Tab-PE's own eval uses, for apples-to-apples comparison.
    from tabicl import TabICLClassifier
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score

    feature_columns = [c for c in train_df.columns if c != label_column]
    syn_X, test_X = synth_df[feature_columns].copy(), test_df[feature_columns].copy()
    syn_y, test_y = synth_df[label_column].copy(), test_df[label_column].copy()

    for col in feature_columns:
        if col in cat_columns:
            enc = LabelEncoder()
            merged = pd.concat([syn_X[col], test_X[col]])
            enc.fit(merged)
            syn_X[col] = enc.transform(syn_X[col])
            test_X[col] = enc.transform(test_X[col])
        else:
            scaler = MinMaxScaler()
            merged = pd.concat([syn_X[col], test_X[col]]).values.reshape(-1, 1)
            scaler.fit(merged)
            syn_X[col] = scaler.transform(syn_X[col].values.reshape(-1, 1))
            test_X[col] = scaler.transform(test_X[col].values.reshape(-1, 1))

    label_enc = LabelEncoder()
    merged_y = pd.concat([syn_y, test_y])
    label_enc.fit(merged_y)
    syn_y_enc = label_enc.transform(syn_y)
    test_y_enc = label_enc.transform(test_y)

    clf = TabICLClassifier()
    clf.fit(syn_X.values, syn_y_enc)
    pred = clf.predict(test_X.values)
    acc = accuracy_score(test_y_enc, pred) * 100
    f1 = f1_score(test_y_enc, pred, average="macro") * 100
    print(f"AIM synthetic data -> TabICL test accuracy: {acc:.2f}%, macro F1: {f1:.2f}")

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump({"test_accuracy": acc, "test_macro_f1": f1}, f, indent=2)
