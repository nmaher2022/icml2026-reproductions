"""Resume-only helper: the AIM mechanism run for person_activity already completed
(results/person_activity_aim/synthetic.csv + timing.json exist, 1170.7s) but the background job
was killed mid-way during the TabICL evaluation step (session teardown, not a script bug -- see
REPRO_LOG.md). This just redoes the evaluation half of aim_baseline.py against the
already-generated synthetic.csv, so we don't have to redo the expensive 1170s AIM mechanism run.

Deviation from aim_baseline.py's original eval code, disclosed: AIM's `num_synth_rows` defaulted
to the full training set size (115402 rows), but Tab-PE's own person_activity.py evaluates its
classifier on only 5000 synthetic rows per iteration (official script default). Subsampling AIM's
output to 5000 rows here for an apples-to-apples same-sample-budget comparison, and because
TabICL's in-context-learning cost is unclear/possibly infeasible at 115K "training" rows.
"""

import json
import os

import pandas as pd

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "person_activity_aim")

base = "https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular/real/person-activity/"
train_df = pd.read_csv(f"{base}person-activity_train.csv")
test_df = pd.read_csv(f"{base}person-activity_test.csv")
import urllib.request
meta = json.load(urllib.request.urlopen(f"{base}person-activity_metadata.json"))

cat_columns = meta["cat_columns"]
label_column = meta["label_columns"][0]

synth_df = pd.read_csv(os.path.join(out_dir, "synthetic.csv"))
SAMPLE_BUDGET = 5000  # matches Tab-PE's own person_activity.py num_samples budget
if len(synth_df) > SAMPLE_BUDGET:
    synth_df = synth_df.sample(n=SAMPLE_BUDGET, random_state=42).reset_index(drop=True)

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

print(f"Fitting TabICL on {len(syn_X)} AIM-synthetic rows, evaluating on {len(test_X)} test rows...")
clf = TabICLClassifier()
clf.fit(syn_X.values, syn_y_enc)
pred = clf.predict(test_X.values)
acc = accuracy_score(test_y_enc, pred) * 100
f1 = f1_score(test_y_enc, pred, average="macro") * 100
print(f"AIM synthetic data (person_activity) -> TabICL test accuracy: {acc:.2f}%, macro F1: {f1:.2f}")

with open(os.path.join(out_dir, "eval.json"), "w") as f:
    json.dump({"test_accuracy": acc, "test_macro_f1": f1}, f, indent=2)
print("Saved eval.json")
