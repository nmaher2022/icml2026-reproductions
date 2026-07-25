# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly", "pandas", "numpy"]
# ///
"""Figures for the DFN reproduction logbook (dataviz palette, cdn plotly)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
OUT = HERE
REPO = HERE / "deep-flow-networks" / "ICML_experiments"

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
SURF = "#fcfcfb"
MODEL_COLOR = {"DFN": BLUE, "MLP": ORANGE, "LSET": AQUA}

LAYOUT = dict(template="none", paper_bgcolor=SURF, plot_bgcolor=SURF,
              font=dict(color="#0b0b0b", size=13),
              margin=dict(l=60, r=30, t=60, b=60))


def fig_claim2():
    frames = []
    for label, path in [("Quadratic n=16 (official artifacts)", REPO / "main_text/outputs/quadratic/quadratic_summary.csv"),
                        ("Resource alloc. (official artifacts)", REPO / "main_text/outputs/resource_allocation/resource_allocation_summary.csv"),
                        ("MDVSP (official artifacts)", REPO / "main_text/outputs/mdvsp/mdvsp_summary.csv"),
                        ("Quadratic n=8 (FRESH retrain)", HERE / "outputs_small/quadratic_small/quadratic_small_summary.csv")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["experiment"] = label
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "claim2_fig_raw.csv", index=False)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Test MSE (normalized, log scale)",
                                                        "Integer-optimization time, s (log scale)"))
    exps = list(dict.fromkeys(df["experiment"]))
    xpos = {e: i for i, e in enumerate(exps)}
    offs = {"DFN": -0.22, "MLP": 0.0, "LSET": 0.22}
    for model in ["DFN", "MLP", "LSET"]:
        sub = df[df["model"] == model]
        xs = [xpos[e] + offs[model] for e in sub["experiment"]]
        for col, metric, se in ((1, "test_mse_norm_mean", "test_mse_norm_se"),
                                (2, "solve_time_s_mean", "solve_time_s_se")):
            fig.add_trace(go.Bar(
                x=xs, y=sub[metric], width=0.2, name=model, legendgroup=model,
                showlegend=(col == 1), marker=dict(color=MODEL_COLOR[model],
                                                   line=dict(color=SURF, width=2)),
                error_y=dict(type="data", array=sub[se], color=GRAY, thickness=1.5),
                customdata=np.stack([sub["experiment"], sub["n_success"], sub["n_runs"]], -1),
                hovertemplate=("%{customdata[0]}<br>" + metric + "=%{y:.4g}"
                               "<br>runs ok=%{customdata[1]}/%{customdata[2]}"
                               "<extra>" + model + "</extra>")), row=1, col=col)
    for ax in ("xaxis", "xaxis2"):
        fig.update_layout({ax: dict(tickvals=list(range(len(exps))),
                                    ticktext=[e.replace(" (", "<br>(") for e in exps],
                                    tickfont=dict(size=10), showgrid=False)})
    for ax in ("yaxis", "yaxis2"):
        fig.update_layout({ax: dict(type="log", gridcolor="#e8e7e3")})
    fig.update_layout(**LAYOUT, barmode="overlay",
                      title="Claim 2 — accuracy and integer-optimization speed (mean ± s.e.)",
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    fig.write_html(OUT / "claim2_dfn_fig.html", include_plotlyjs="cdn")
    print("wrote claim2_dfn_fig.html")


def fig_claim1():
    df = pd.read_csv(HERE / "claim1_universality.csv")
    df.to_csv(OUT / "claim1_fig_raw.csv", index=False)
    fig = go.Figure()
    small = df[(df["layers"].str.contains("20")) & (~df["fn"].str.startswith("CONTROL"))]
    big = df[(df["layers"].str.contains("60")) & (~df["fn"].str.startswith("CONTROL"))]
    ctrl = df[df["fn"].str.startswith("CONTROL")]
    for label, sub, color in [("small DFN [4,20,4]", small, ORANGE),
                              ("large DFN [12,60,12]", big, BLUE),
                              ("large DFN — CONTROL (no convex ext.)", ctrl, GRAY)]:
        fig.add_trace(go.Bar(
            x=sub["fn"], y=sub["max_rel_err"], name=label,
            marker=dict(color=color, line=dict(color=SURF, width=2)),
            customdata=np.stack([sub["mean_rel_err"], sub["midpoint_violations"]], -1),
            hovertemplate=("%{x}: max err/range=%{y:.4f}<br>mean=%{customdata[0]:.4f}"
                           "<br>midpoint-convexity violations=%{customdata[1]}"
                           "<extra>" + label + "</extra>")))
    fig.update_layout(**LAYOUT, barmode="group",
                      title="Claim 1 — trained-DFN max |error| / range over the full grid {-7..7}² (SGD-trained)",
                      xaxis=dict(title="ground-truth function", showgrid=False),
                      yaxis=dict(title="max abs error / range(g)", gridcolor="#e8e7e3"),
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    fig.write_html(OUT / "claim1_dfn_fig.html", include_plotlyjs="cdn")
    print("wrote claim1_dfn_fig.html")


def fig_aevo():
    df = pd.read_csv(REPO / "appendix/outputs/a_evolution/a_evolution_summary.csv")
    df.to_csv(OUT / "aevo_fig_raw.csv", index=False)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("log √det(A) over training", "re-solve time (s, log scale)"))
    fig.add_trace(go.Scatter(x=df["checkpoint_epoch"], y=df["A_log_pdet_mean"],
                             mode="lines+markers", name="log √det(A)",
                             error_y=dict(type="data", array=df["A_log_pdet_se"], color=GRAY),
                             line=dict(color=BLUE, width=2), marker=dict(size=8),
                             hovertemplate="epoch %{x}: log√det(A)=%{y:.1f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df["checkpoint_epoch"], y=df["solve_time_s_mean"],
                             mode="lines+markers", name="re-solve time (s)",
                             error_y=dict(type="data", array=df["solve_time_s_se"], color=GRAY),
                             line=dict(color=ORANGE, width=2), marker=dict(size=8),
                             hovertemplate="epoch %{x}: solve=%{y:.2f}s<extra></extra>"),
                  row=2, col=1)
    fig.update_layout(**LAYOUT,
                      title="A-drift vs solve time (official artifacts): ∆(A) governs optimization cost",
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    fig.update_xaxes(title_text="training epoch checkpoint", row=2, col=1, gridcolor="#e8e7e3")
    fig.update_yaxes(row=1, col=1, gridcolor="#e8e7e3")
    fig.update_yaxes(row=2, col=1, type="log", gridcolor="#e8e7e3")
    fig.write_html(OUT / "aevo_dfn_fig.html", include_plotlyjs="cdn")
    print("wrote aevo_dfn_fig.html")


if __name__ == "__main__":
    fig_claim2()
    if (HERE / "claim1_universality.csv").exists():
        fig_claim1()
    fig_aevo()
