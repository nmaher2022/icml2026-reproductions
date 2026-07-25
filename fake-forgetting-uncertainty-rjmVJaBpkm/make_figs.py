#!/usr/bin/env python3
"""Generate poster/logbook figures from ./outputs/results.json (matplotlib PNGs)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
d = json.load(open("./outputs/results.json"))
A = "0.1"
os.makedirs("figs", exist_ok=True)
ACC = "#2563eb"; EMPH = "#d97706"; GREY = "#94a3b8"

def cr(k, which="forget"): return d[k]["per_alpha"][A][which]["cr"]

# --- Fig 1: fake-forgetting recover ratio (Claim 2) ---
meth = [("retrain","RT"),("finetune","FT"),("RL","RL")]
ratios = [d[k]["per_alpha"][A]["forget"]["recover_ratio"]*100 for k,_ in meth]
paper = [30.6,58.3,45.5]
x=np.arange(len(meth)); w=0.38
fig,ax=plt.subplots(figsize=(5,3.2))
ax.bar(x-w/2, ratios, w, label="this repro (toy)", color=ACC)
ax.bar(x+w/2, paper, w, label="paper", color=GREY)
ax.set_xticks(x); ax.set_xticklabels([n for _,n in meth])
ax.set_ylabel("recover ratio (%)"); ax.set_ylim(0,105)
ax.set_title("Claim 2: fake forgetting — GT still in conformal set")
ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3)
for i,v in enumerate(ratios): ax.text(i-w/2,v+1,f"{v:.0f}",ha="center",fontsize=8)
fig.tight_layout(); fig.savefig("figs/fig_claim2.png",dpi=150); plt.close(fig)

# --- Fig 2: CR_Df vs CR_Dtest across methods (Claim 1/3) ---
order=[("retrain","RT"),("finetune","FT"),("RL","RL"),("GA","GA"),("teacher","Tch"),
       ("ssd","SSD"),("ga_plus","NG+"),("salun","Sal")]
crf=[cr(k,"forget") for k,_ in order]; crt=[cr(k,"test") for k,_ in order]
x=np.arange(len(order)); w=0.4
fig,ax=plt.subplots(figsize=(6,3.2))
ax.bar(x-w/2, crf, w, label="CR $\\mathcal{D}_f$ (↓ = forget)", color=EMPH)
ax.bar(x+w/2, crt, w, label="CR $\\mathcal{D}_{test}$ (↑ = utility)", color=ACC)
ax.set_xticks(x); ax.set_xticklabels([n for _,n in order]); ax.set_ylabel("CR")
ax.set_title("Claim 3: CR ranks methods (Teacher: low CR_Dtest = destroyed utility)")
ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3)
fig.tight_layout(); fig.savefig("figs/fig_claim3.png",dpi=150); plt.close(fig)

# --- Fig 3: CPU UA improvement (Claim 5) ---
labels=["CPU-FT","CPU-RL"]
ua0=[d["cpu-finetune-l0.0"]["UA"]*100, d["cpu-RL-l0.0"]["UA"]*100]
ua5=[d["cpu-finetune-l0.5"]["UA"]*100, d["cpu-RL-l0.5"]["UA"]*100]
ta0=[d["cpu-finetune-l0.0"]["TA"]*100, d["cpu-RL-l0.0"]["TA"]*100]
ta5=[d["cpu-finetune-l0.5"]["TA"]*100, d["cpu-RL-l0.5"]["TA"]*100]
x=np.arange(len(labels)); w=0.35
fig,ax=plt.subplots(figsize=(5,3.2))
ax.bar(x-w/2, ua0, w, label="UA λ=0", color=GREY)
ax.bar(x-w/2, [b-a for a,b in zip(ua0,ua5)], w, bottom=ua0, label="UA gain (λ=0.5)", color=EMPH)
ax.plot(x+w/2, ta0, "o--", color=ACC, label="TA λ=0")
ax.plot(x+w/2, ta5, "s-", color=ACC, label="TA λ=0.5")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("accuracy (%)")
ax.set_title("Claim 5: CPU (λ=0.5) raises UA, TA ~flat")
ax.legend(fontsize=7); ax.grid(axis="y",alpha=.3); ax.set_ylim(0,100)
fig.tight_layout(); fig.savefig("figs/fig_claim5.png",dpi=150); plt.close(fig)
print("wrote figs/fig_claim2.png figs/fig_claim3.png figs/fig_claim5.png")
