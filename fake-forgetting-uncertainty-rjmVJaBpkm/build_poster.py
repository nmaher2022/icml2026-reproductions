#!/usr/bin/env python3
"""Build a self-contained reproduction poster: poster.html -> poster.png -> poster_embed.html."""
import base64, json, pathlib
from playwright.sync_api import sync_playwright

d = json.load(open("./outputs/results.json"))
def b64(p): return base64.b64encode(pathlib.Path(p).read_bytes()).decode()
fig2, fig3, fig5 = (f"data:image/png;base64,{b64(f)}" for f in
                    ["figs/fig_claim2.png","figs/fig_claim3.png","figs/fig_claim5.png"])

# key numbers
cf0,cf5 = d["cpu-finetune-l0.0"], d["cpu-finetune-l0.5"]

POSTER = f"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{{--acc:#2563eb;--accd:#1e3a8a;--emph:#d97706;--ink:#0f172a;--mut:#475569;
--card:#ffffff;--tint:#f1f5f9;--bd:#e2e8f0;--bg:#f8fafc;}}
*{{margin:0;box-sizing:border-box;font-family:'DejaVu Sans',Arial,sans-serif}}
.poster{{width:1080px;background:var(--bg);padding:40px;color:var(--ink)}}
.hdr{{border-left:10px solid var(--acc);padding:6px 0 6px 22px;margin-bottom:20px}}
.badge{{display:inline-block;background:var(--acc);color:#fff;font-weight:700;font-size:13px;
padding:4px 12px;border-radius:20px;letter-spacing:.5px}}
h1{{font-size:33px;line-height:1.15;margin:10px 0 6px}}
h1 .a{{color:var(--acc)}}
.sub{{font-size:16px;color:var(--mut);line-height:1.4}}
.meta{{margin-top:10px;font-size:13px;color:var(--mut)}}
.meta b{{color:var(--accd)}}
.outcome{{background:#ecfdf5;border:1px solid #6ee7b7;border-left:8px solid #059669;
border-radius:10px;padding:14px 18px;margin:16px 0;font-size:15px;line-height:1.45}}
.outcome b{{color:#065f46}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px 18px;
box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.card.full{{grid-column:1/3}}
.st{{display:flex;align-items:center;gap:10px;font-weight:800;color:var(--accd);font-size:17px;margin-bottom:8px}}
.num{{width:26px;height:26px;border-radius:50%;background:var(--acc);color:#fff;display:flex;
align-items:center;justify-content:center;font-size:14px;flex-shrink:0}}
.card p{{font-size:13.5px;line-height:1.4;color:var(--ink)}}
.k{{color:var(--acc);font-weight:700}} .e{{color:var(--emph);font-weight:700}}
img.fig{{width:100%;border:1px solid var(--bd);border-radius:8px;margin-top:8px;background:#fff}}
.pill{{display:inline-block;background:var(--tint);border-radius:6px;padding:1px 8px;font-weight:700;font-size:12.5px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}}
th,td{{border:1px solid var(--bd);padding:4px 7px;text-align:center}}
th{{background:var(--tint);color:var(--accd)}}
.foot{{margin-top:18px;background:#fff7ed;border:1px solid #fdba74;border-left:8px solid var(--emph);
border-radius:10px;padding:12px 16px;font-size:12.5px;line-height:1.4;color:#7c2d12}}
.foot b{{color:#9a3412}}
.links{{margin-top:12px;font-size:12px;color:var(--mut);text-align:center}}
</style></head><body><div class="poster">

<div class="hdr">
  <span class="badge">ICML 2026 REPRODUCTION · #2446</span>
  <h1>Tackling Fake Forgetting through <span class="a">Uncertainty Quantification</span></h1>
  <div class="sub">Conformal prediction exposes that unlearned models still "remember" — a CPU toy reproduction of the official code.</div>
  <div class="meta"><b>Paper:</b> arXiv:2501.19403 · <b>Data/Model:</b> CIFAR-10 subset + ResNet-18 ·
  <b>Compute:</b> 8-core CPU (no GPU; HF Jobs 402) · ~4 h · ~$0</div>
</div>

<div class="outcome"><b>Outcome — mechanism reproduced.</b> The CR / MIACR / CPU-loss code is verified
<b>12/12</b> against the paper's equations. The original ResNet-18 memorized the data
(forget-acc {d['original']['forget_acc']:.2f}, TA {d['original']['TA']:.2f}); every claim's mechanism holds at toy scale.
Exact paper magnitudes, the FF method, and the Tiny-ImageNet half of Claim 5 need a GPU.</div>

<div class="grid">
  <div class="card">
    <div class="st"><span class="num">2</span>Fake forgetting is real</div>
    <p>Among forget points the model <span class="e">misclassifies</span> (UA calls them "forgotten"),
    a large fraction still have the true label <span class="k">inside the conformal set</span> —
    even gold-standard retrain: <span class="pill">79%</span>.</p>
    <img class="fig" src="{fig2}">
  </div>
  <div class="card">
    <div class="st"><span class="num">1·3</span>CR &gt; UA</div>
    <p>CR = Coverage / Set Size. Teacher tops UA (92.5%) &amp; CR<sub>Df</sub> but its
    <span class="k">CR<sub>Dtest</sub> collapses to 0.10</span> — utility destroyed, invisible to UA.
    Dtest coverage ≈ 0.90 confirms valid calibration.</p>
    <img class="fig" src="{fig3}">
  </div>
  <div class="card">
    <div class="st"><span class="num">4</span>MIA is unreliable</div>
    <p>Traditional MIA and <span class="k">MIACR invert</span>: Teacher looks best under MIA
    (<span class="pill">0.00</span>) but worst under MIACR (<span class="pill">0.039</span>);
    FT/SSD/NegGrad+ score badly on MIA but 0.000 on MIACR.</p>
    <table><tr><th>Method</th><th>MIA↓</th><th>MIACR</th></tr>
    <tr><td>Teacher</td><td>0.00</td><td>0.039</td></tr>
    <tr><td>Salun</td><td>0.15</td><td>0.031</td></tr>
    <tr><td>SSD</td><td>0.96</td><td>0.000</td></tr>
    <tr><td>NegGrad+</td><td>0.81</td><td>0.000</td></tr></table>
  </div>
  <div class="card">
    <div class="st"><span class="num">5</span>CPU improves forgetting</div>
    <p>Adding the conformal loss (λ=0.5) <span class="e">raises UA</span> while TA barely moves &amp;
    CR<sub>Dtest</sub> is unchanged. CPU-FT: UA {cf0['UA']*100:.0f}→{cf5['UA']*100:.0f}%,
    TA {cf0['TA']*100:.0f}→{cf5['TA']*100:.0f}%.</p>
    <img class="fig" src="{fig5}">
  </div>
</div>

<div class="foot"><b>⚠ GPU required for full credit.</b> This is a CPU toy (HF Jobs returned 402, no GPU).
Not fully credited: paper-scale magnitudes (Claims 2–5), the FF method (Claim 3), and the entire
Tiny-ImageNet + ViT half of Claim 5. The <b>same parameterized script</b> re-runs at scale via
<code>repro_toy.py --device cuda --n_per_class 5000 --orig_epochs 200</code>.</div>

<div class="links">github.com/TIML-Group/Conformal-Prediction-Unlearning · arxiv.org/abs/2501.19403 ·
Space: nmaher/repro-tackling-fake-forgetting-through-uncertainty-quantification</div>

</div></body></html>"""

pathlib.Path("poster.html").write_text(POSTER)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(device_scale_factor=2)
    pg.goto("file://" + str(pathlib.Path("poster.html").resolve()))
    pg.wait_for_timeout(400)
    el = pg.query_selector(".poster")
    el.screenshot(path="poster.png")
    b.close()

poster_uri = f"data:image/png;base64,{b64('poster.png')}"
EMBED = f"""<div style="max-width:100%;text-align:center">
<img src="{poster_uri}" alt="Reproduction poster: Tackling Fake Forgetting through Uncertainty Quantification"
style="max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:8px"/>
</div>"""
pathlib.Path("poster_embed.html").write_text(EMBED)
print("wrote poster.html, poster.png, poster_embed.html")
print("poster.png bytes:", pathlib.Path("poster.png").stat().st_size)
