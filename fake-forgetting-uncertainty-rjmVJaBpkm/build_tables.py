#!/usr/bin/env python3
"""Emit per-claim markdown tables from ./outputs/results.json for the logbook."""
import json, sys
d = json.load(open("./outputs/results.json"))
A = "0.1"  # alpha for the headline tables (challenge Claim 3 uses alpha=0.1)

# display order + pretty names (paper's method labels)
ORDER = [("retrain","RT"),("finetune","FT"),("RL","RL"),("GA","GA"),("teacher","Teacher"),
         ("ssd","SSD"),("ga_plus","NegGrad+"),("salun","Salun")]
def has(k): return k in d and not (isinstance(d[k],dict) and "error" in d[k])
def fa(k,which="forget"): return d[k]["per_alpha"][A][which]

o = d["original"]
print("### ORIGINAL")
print(f"forget_acc={o['forget_acc']:.3f} RA={o['RA']:.3f} TA={o['TA']:.3f} UA={o['UA']:.3f}\n")

# ---- Claim 2: fake forgetting (RT/FT/RL) ----
print("### CLAIM 2 — fake forgetting recover ratio (alpha=0.1)")
print("| Method | UA (%) | n_mislabel | n_in_set | recover ratio (%) | paper (10% forget) |")
print("| --- | --- | --- | --- | --- | --- |")
paper2 = {"retrain":"30.6","finetune":"58.3","RL":"45.5"}
for k,name in [("retrain","RT (retrain)"),("finetune","FT (finetune)"),("RL","RL (random label)")]:
    f=fa(k); print(f"| {name} | {d[k]['UA']*100:.1f} | {f['n_mislabel']} | {f['n_in_set']} | {f['recover_ratio']*100:.1f} | {paper2[k]} |")

# ---- Claim 3: CR eval across methods ----
print("\n### CLAIM 3 — Coverage / Set Size / CR (alpha=0.1)")
print("| Method | UA (%) | Cov Df | Size Df | CR Df ↓ | Cov Dtest | Size Dtest | CR Dtest ↑ |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for k,name in ORDER:
    if not has(k): print(f"| {name} | — diverged (NaN) — | | | | | | |"); continue
    f=fa(k); t=fa(k,"test")
    print(f"| {name} | {d[k]['UA']*100:.1f} | {f['coverage']:.2f} | {f['set_size']:.2f} | {f['cr']:.3f} | {t['coverage']:.2f} | {t['set_size']:.2f} | {t['cr']:.3f} |")

# ---- Claim 4: MIA vs MIACR ----
print("\n### CLAIM 4 — traditional MIA vs MIACR")
print("| Method | MIA (traditional) | MIACR | UA (%) | CR Df |")
print("| --- | --- | --- | --- | --- |")
for k,name in ORDER:
    if not has(k): print(f"| {name} | — | — | — | — |"); continue
    mia=d[k]["MIA_traditional"]; mc=d[k]["MIACR"].get("miacr",float('nan'))
    print(f"| {name} | {mia:.2f} | {mc:.3f} | {d[k]['UA']*100:.1f} | {fa(k)['cr']:.3f} |")

# ---- Claim 5: CPU ----
print("\n### CLAIM 5 — CPU (lambda 0 vs 0.5)")
print("| Method | UA λ0 | UA λ0.5 | ΔUA | TA λ0 | TA λ0.5 | ΔTA | CR_Df λ0 | CR_Df λ0.5 | CR_Dt λ0 | CR_Dt λ0.5 |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
dua=[]; dta=[]
for base,name in [("finetune","CPU-FT"),("RL","CPU-RL")]:
    a=d[f"cpu-{base}-l0.0"]; b=d[f"cpu-{base}-l0.5"]
    ua0,ua5=a["UA"]*100,b["UA"]*100; ta0,ta5=a["TA"]*100,b["TA"]*100
    dua.append(ua5-ua0); dta.append(ta5-ta0)
    print(f"| {name} | {ua0:.1f} | {ua5:.1f} | {ua5-ua0:+.1f} | {ta0:.1f} | {ta5:.1f} | {ta5-ta0:+.1f} | "
          f"{a['per_alpha'][A]['forget']['cr']:.3f} | {b['per_alpha'][A]['forget']['cr']:.3f} | "
          f"{a['per_alpha'][A]['test']['cr']:.3f} | {b['per_alpha'][A]['test']['cr']:.3f} |")
print(f"| **avg** | | | **{sum(dua)/len(dua):+.1f}** | | | **{sum(dta)/len(dta):+.1f}** | | | | |")
print(f"\nPaper (CIFAR-10, lambda=0.5): avg ΔUA +3.93%, avg ΔTA -1.0%.")
