# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly", "pandas", "numpy"]
# ///
"""Figures for the Gaussian-mechanism reproduction logbook."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
SURF = "#fcfcfb"
LAYOUT = dict(template="none", paper_bgcolor=SURF, plot_bgcolor=SURF,
              font=dict(color="#0b0b0b", size=13),
              margin=dict(l=70, r=30, t=60, b=60))


def fig_claim1():
    df = pd.read_csv(HERE / "claim1_asymptotic.csv")
    ins = df[df["regime"].str.startswith("inside")]
    fig = go.Figure()
    comps = [("SGG(p=1.0)", BLUE), ("SGG(p=1.3)", ORANGE), ("SGG(p=3.0)", AQUA)]
    for name, color in comps:
        fig.add_trace(go.Scatter(
            x=ins["T"], y=ins[name] / 1e-3, mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=8),
            hovertemplate="T=%{x}: delta ratio=%{y:.3f}<extra>" + name + "</extra>"))
    fig.add_hline(y=1.0, line=dict(color=GRAY, dash="dash", width=2),
                  annotation_text="Gaussian benchmark (Thm 3.1 asymptote)",
                  annotation_font_color=GRAY)
    fig.update_layout(**LAYOUT,
                      title="Claim 1 — competitor δ(ε)/δ_Gauss at equal MSE budget (ε=1, δ=10⁻³)",
                      xaxis=dict(type="log", title="dimension T", gridcolor="#e8e7e3"),
                      yaxis=dict(title="δ_competitor / δ_Gaussian", gridcolor="#e8e7e3",
                                 rangemode="tozero"),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig.write_html(HERE / "claim1_gauss_fig.html", include_plotlyjs="cdn")
    ins.to_csv(HERE / "claim1_fig_raw.csv", index=False)
    print("wrote claim1_gauss_fig.html")


def fig_claim45():
    f4 = pd.read_csv(HERE / "claim4_fig2.csv")
    f5 = pd.read_csv(HERE / "claim5_table2.csv")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        subplot_titles=("Fig. 2 reproduction: SGG MSE gain (ε=0.1)",
                                        "Table 2 reproduction: δ⋆ lower bounds"))
    x = np.arange(len(f4))
    fig.add_trace(go.Bar(x=x - 0.17, y=f4["gain_pct_ours"], width=0.3, name="our audit",
                         marker=dict(color=BLUE, line=dict(color=SURF, width=2)),
                         hovertemplate="T=%{customdata}: ours %{y:.1f}%<extra></extra>",
                         customdata=f4["T"]), row=1, col=1)
    fig.add_trace(go.Bar(x=x + 0.17, y=f4["gain_pct_paper"], width=0.3, name="paper",
                         marker=dict(color=ORANGE, line=dict(color=SURF, width=2)),
                         hovertemplate="T=%{customdata}: paper %{y:.1f}%<extra></extra>",
                         customdata=f4["T"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=f5["eps"], y=f5["delta_star_ours"], mode="lines+markers",
                             name="ours (mpmath)", line=dict(color=BLUE, width=2),
                             marker=dict(size=9),
                             hovertemplate="ε=%{x}: δ⋆=%{y:.6f}<extra>ours</extra>"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=f5["eps"], y=f5["delta_star_paper"], mode="markers",
                             name="paper Table 2", marker=dict(size=13, symbol="circle-open",
                             color=ORANGE, line=dict(width=2)),
                             hovertemplate="ε=%{x}: δ⋆=%{y:.6f}<extra>paper</extra>"),
                  row=1, col=2)
    fig.update_layout(**LAYOUT, barmode="overlay",
                      title="Claims 4 + 5 — paper numbers vs independent audit",
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    fig.update_xaxes(tickvals=list(x), ticktext=[f"T={t}" for t in f4["T"]],
                     row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="MSE reduction vs best baseline (%)", row=1, col=1,
                     gridcolor="#e8e7e3")
    fig.update_xaxes(type="log", title_text="ε", row=1, col=2, gridcolor="#e8e7e3")
    fig.update_yaxes(title_text="δ⋆", row=1, col=2, gridcolor="#e8e7e3")
    fig.write_html(HERE / "claim45_gauss_fig.html", include_plotlyjs="cdn")
    pd.concat([f4, f5], axis=1).to_csv(HERE / "claim45_fig_raw.csv", index=False)
    print("wrote claim45_gauss_fig.html")


def fig_claim6():
    df = pd.read_csv(HERE / "claim6_composition.csv")
    g = df.dropna(subset=["d_analytic"]) if "d_analytic" in df else pd.DataFrame()
    l2 = df.dropna(subset=["d_mc"]) if "d_mc" in df else pd.DataFrame()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Gaussian member: FFT vs closed form",
                                        "ℓ2 mechanism: FFT vs Monte-Carlo"))
    if len(g):
        for eps, color in [(1.0, BLUE), (3.0, ORANGE)]:
            sub = g[g["eps"] == eps]
            fig.add_trace(go.Scatter(x=sub["k"], y=abs(sub["rel_err"]),
                                     mode="lines+markers", name=f"|rel err| ε={eps}",
                                     line=dict(color=color, width=2), marker=dict(size=8),
                                     hovertemplate="k=%{x}: |rel err|=%{y:.2e}<extra></extra>"),
                          row=1, col=1)
    if len(l2):
        for k, color in [(2, BLUE), (4, ORANGE)]:
            sub = l2[l2["k"] == k]
            fig.add_trace(go.Scatter(x=sub["eps_tot"], y=sub["z_score"],
                                     mode="lines+markers", name=f"|z| k={k}",
                                     line=dict(color=color, width=2, dash="dot"),
                                     marker=dict(size=8),
                                     hovertemplate="ε_tot=%{x}: z=%{y:.2f}<extra></extra>"),
                          row=1, col=2)
    fig.update_layout(**LAYOUT, title="Claim 6 — composition accountant tightness",
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0))
    fig.update_xaxes(title_text="compositions k", row=1, col=1, gridcolor="#e8e7e3")
    fig.update_yaxes(type="log", title_text="|relative error|", row=1, col=1, gridcolor="#e8e7e3")
    fig.update_xaxes(title_text="total ε", row=1, col=2, gridcolor="#e8e7e3")
    fig.update_yaxes(title_text="|z| vs 4M-sample MC", row=1, col=2, gridcolor="#e8e7e3",
                     rangemode="tozero")
    fig.write_html(HERE / "claim6_gauss_fig.html", include_plotlyjs="cdn")
    print("wrote claim6_gauss_fig.html")


if __name__ == "__main__":
    fig_claim1()
    fig_claim45()
    fig_claim6()
