# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly", "pandas", "numpy", "kaleido"]
# ///
"""Figures for the Gluon (Muon-to-Gluon LMO optimizers) reproduction logbook."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, GRAY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#c0392b"
SURF = "#fcfcfb"
LAYOUT = dict(template="none", paper_bgcolor=SURF, plot_bgcolor=SURF,
              font=dict(color="#0b0b0b", size=13),
              margin=dict(l=70, r=30, t=60, b=60))


def fig_claim1():
    df = pd.read_csv(HERE / "claim1_special_cases.csv")
    g = df.groupby("special_case")["max_abs_diff"].agg(["max", "count"]).reset_index()
    g = g.sort_values("special_case")
    floor = 1e-17  # so exact-zero bars are still visible on a log axis
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g["special_case"], y=g["max"].clip(lower=floor), width=0.6,
        marker=dict(color=BLUE, line=dict(color=SURF, width=1.5)),
        text=[f"{v:.1e}" for v in g["max"]], textposition="outside",
        customdata=g["count"],
        hovertemplate="%{x}: worst diff=%{customdata} comparisons<extra></extra>"))
    fig.add_hline(y=1e-8, line=dict(color=RED, dash="dash", width=2),
                  annotation_text="pass tolerance (1e-8)", annotation_font_color=RED)
    fig.update_layout(**LAYOUT,
                      title="Claim 1 — general LMO vs. closed-form special cases "
                            "(520/520 comparisons, worst diff per case)",
                      xaxis=dict(title=None, tickangle=-25),
                      yaxis=dict(title="max|general_lmo - closed_form|", type="log",
                                 range=[-17, -6], gridcolor="#e8e7e3"))
    fig.write_html(HERE / "claim1_fig.html", include_plotlyjs="cdn")
    fig.write_image(HERE / "claim1_fig.png", width=1000, height=560, scale=2)
    g.to_csv(HERE / "claim1_fig_raw.csv", index=False)
    print("wrote claim1_fig.html / .png")


def fig_claim23():
    c2 = pd.read_csv(HERE / "claim2_deterministic_rate.csv")
    c3 = pd.read_csv(HERE / "claim3_stochastic_rate.csv")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Claim 2 — deterministic rate (claimed slope -0.5)",
                                        "Claim 3 — stochastic rate (claimed slope -0.25)"))

    def add_panel(df, col, slope, intercept, claimed_slope, name):
        fig.add_trace(go.Scatter(x=df["K"], y=df["metric_value"], mode="markers+lines",
                                 name=f"{name} (measured)", line=dict(color=BLUE, width=2),
                                 marker=dict(size=9),
                                 hovertemplate="K=%{x}: metric=%{y:.4f}<extra></extra>"),
                      row=1, col=col)
        K = np.array(sorted(df["K"]))
        fit = np.exp(intercept) * K.astype(float) ** slope
        fig.add_trace(go.Scatter(x=K, y=fit, mode="lines", name=f"fit slope={slope:.2f}",
                                 line=dict(color=ORANGE, width=2, dash="dot")),
                      row=1, col=col)
        ref = df["metric_value"].iloc[0] * (K / K[0]).astype(float) ** claimed_slope
        fig.add_trace(go.Scatter(x=K, y=ref, mode="lines",
                                 name=f"claimed slope={claimed_slope:.2f}",
                                 line=dict(color=GRAY, width=2, dash="dash")),
                      row=1, col=col)

    add_panel(c2, 1, -1.199970, 4.542459, -0.5, "claim 2")
    add_panel(c3, 2, -0.346234, 2.386075, -0.25, "claim 3")
    fig.update_layout(**LAYOUT, title="Claims 2 + 3 — convergence-rate log-log fits vs. claimed rates",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig.update_xaxes(type="log", title_text="K", row=1, col=1, gridcolor="#e8e7e3")
    fig.update_xaxes(type="log", title_text="K", row=1, col=2, gridcolor="#e8e7e3")
    fig.update_yaxes(type="log", title_text="weighted dual-norm metric", row=1, col=1,
                     gridcolor="#e8e7e3")
    fig.update_yaxes(type="log", title_text="weighted dual-norm metric", row=1, col=2,
                     gridcolor="#e8e7e3")
    fig.write_html(HERE / "claim23_fig.html", include_plotlyjs="cdn")
    fig.write_image(HERE / "claim23_fig.png", width=1200, height=560, scale=2)
    pd.concat([c2.assign(claim=2), c3.assign(claim=3)], ignore_index=True).to_csv(
        HERE / "claim23_fig_raw.csv", index=False)
    print("wrote claim23_fig.html / .png")


def fig_claim45():
    c4 = pd.read_csv(HERE / "claim4_transformer_smoothness.csv")
    c5 = pd.read_csv(HERE / "claim5_cnn_smoothness.csv")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Claim 4 — NanoGPT toy: L1 by layer group (ours vs. paper)",
                                        "Claim 5 — CNN toy: L1 by layer group (post-fix; "
                                        "paper values as reference lines)"))
    x4 = np.arange(len(c4))
    fig.add_trace(go.Bar(x=x4 - 0.17, y=c4["L1_fit"], width=0.3, name="ours",
                         marker=dict(color=BLUE, line=dict(color=SURF, width=2)),
                         hovertemplate="%{customdata}: ours L1=%{y:.2f}<extra></extra>",
                         customdata=c4["layer_group"]), row=1, col=1)
    paper_l1 = {"transformer_block": 70.0, "embed_output": 1.3}
    fig.add_trace(go.Bar(x=x4 + 0.17, y=[paper_l1[g] for g in c4["layer_group"]], width=0.3,
                         name="paper (approx.)",
                         marker=dict(color=ORANGE, line=dict(color=SURF, width=2)),
                         hovertemplate="%{customdata}: paper L1~=%{y:.2f}<extra></extra>",
                         customdata=c4["layer_group"]), row=1, col=1)
    fig.update_xaxes(tickvals=list(x4), ticktext=list(c4["layer_group"]), row=1, col=1,
                     showgrid=False)
    fig.update_yaxes(title_text="L1_fit", type="log", row=1, col=1, gridcolor="#e8e7e3")

    x5 = np.arange(len(c5))
    fig.add_trace(go.Bar(x=x5, y=c5["L1_fit"], width=0.5, name="ours (per layer)",
                         marker=dict(color=AQUA, line=dict(color=SURF, width=2)),
                         hovertemplate="%{customdata}: ours L1=%{y:.2f}<extra></extra>",
                         customdata=c5["layer_group"], showlegend=False), row=1, col=2)
    fig.add_hline(y=3.0, line=dict(color=ORANGE, dash="dash", width=2), row=1, col=2)
    fig.add_hline(y=0.03, line=dict(color=RED, dash="dash", width=2), row=1, col=2)
    fig.add_annotation(x=0.55, y=np.log10(3.0) + 0.12, xref="x2 domain", yref="y2",
                       text="paper non-head L1~=3", showarrow=False, font=dict(color=ORANGE))
    fig.add_annotation(x=0.55, y=np.log10(0.03) + 0.12, xref="x2 domain", yref="y2",
                       text="paper head L1~=0.03", showarrow=False, font=dict(color=RED))
    fig.update_xaxes(tickvals=list(x5), ticktext=list(c5["layer_group"]), row=1, col=2,
                     showgrid=False)
    fig.update_yaxes(title_text="L1_fit", type="log", range=[-2, 1.1], row=1, col=2,
                     gridcolor="#e8e7e3")

    fig.update_layout(**LAYOUT, barmode="group",
                      title="Claims 4 + 5 — layer-wise smoothness (L1): toy reproduction vs. paper",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig.write_html(HERE / "claim45_fig.html", include_plotlyjs="cdn")
    fig.write_image(HERE / "claim45_fig.png", width=1200, height=560, scale=2)
    pd.concat([c4.assign(claim=4), c5.assign(claim=5)], ignore_index=True).to_csv(
        HERE / "claim45_fig_raw.csv", index=False)
    print("wrote claim45_fig.html / .png")


if __name__ == "__main__":
    fig_claim1()
    fig_claim23()
    fig_claim45()
