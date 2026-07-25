# /// script
# requires-python = ">=3.10"
# dependencies = ["plotly", "pandas", "numpy"]
# ///
"""Figures for the logbook (palette: dataviz default; hover on; cdn plotly)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#52514e"
SURF = "#fcfcfb"

LAYOUT = dict(
    template="none", paper_bgcolor=SURF, plot_bgcolor=SURF,
    font=dict(color="#0b0b0b", size=13),
    margin=dict(l=60, r=30, t=60, b=50),
)


def jitter(rng, n, width=0.18):
    return rng.uniform(-width, width, n)


def fig_ratio(csvs, bound, title, out_html, out_csv):
    rng = np.random.default_rng(0)
    frames = []
    for label, path in csvs:
        df = pd.read_csv(path)
        df["series"] = label
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_csv, index=False)

    cats = list(dict.fromkeys(all_df["oracle"]))
    xmap = {c: i for i, c in enumerate(cats)}
    fig = go.Figure()
    colors = {lbl: c for (lbl, _), c in zip(csvs, [BLUE, ORANGE, AQUA])}
    for label, _ in csvs:
        sub = all_df[all_df["series"] == label]
        # prefer the expectation statistic when present, else single-run min
        y = sub["exp_min_ratio"].fillna(sub["mean_min_ratio"]) \
            if "exp_min_ratio" in sub else sub["min_ratio"]
        x = np.array([xmap[o] for o in sub["oracle"]], dtype=float)
        x = x + jitter(rng, len(x))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=label,
            marker=dict(color=colors[label], size=8, opacity=0.75,
                        line=dict(color=SURF, width=2)),
            customdata=np.stack([sub["oracle"], sub["k"], sub["seed"],
                                 sub["adversarial"]], axis=-1),
            hovertemplate=("oracle=%{customdata[0]} k=%{customdata[1]} "
                           "seed=%{customdata[2]} adv=%{customdata[3]}<br>"
                           "min expected ratio=%{y:.3f}<extra>" + label + "</extra>"),
        ))
    fig.add_hline(y=bound, line=dict(color=GRAY, dash="dash", width=2),
                  annotation_text=f"claimed bound {bound}",
                  annotation_font_color=GRAY)
    fig.update_layout(
        **LAYOUT, title=title,
        xaxis=dict(tickvals=list(range(len(cats))), ticktext=cats,
                   title="objective family", showgrid=False, zeroline=False),
        yaxis=dict(title="min-over-stream approximation ratio (per instance)",
                   rangemode="tozero", gridcolor="#e8e7e3"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print("wrote", out_html)


def fig_scaling(out_html, out_csv):
    rnd = pd.read_csv("claim2_random.csv")
    adv = pd.read_csv("claim2_adversarial.csv")
    rnd["mode"], adv["mode"] = "random deletions", "adversarial deletions"
    both = pd.concat([rnd, adv], ignore_index=True)
    both.to_csv(out_csv, index=False)

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=("vs ground-set size n (k=16, eps=0.2)",
                        "vs cardinality k (n=1000, eps=0.2)"))
    for col, (sweep, xkey) in enumerate([("n", "n"), ("k", "k")], start=1):
        for mode, color in [("random deletions", BLUE),
                            ("adversarial deletions", ORANGE)]:
            sub = both[(both["sweep"] == sweep) & (both["mode"] == mode)]
            g = sub.groupby(xkey)["mean_q"].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=g[xkey], y=g["mean_q"], mode="lines+markers", name=mode,
                showlegend=(col == 1), legendgroup=mode,
                line=dict(color=color, width=2), marker=dict(size=8),
                hovertemplate=xkey + "=%{x}<br>mean queries/update=%{y:.1f}"
                              "<extra>" + mode + "</extra>"),
                row=1, col=col)
        # naive recompute reference n*k
        sub = both[both["sweep"] == sweep]
        g = sub.groupby(xkey)["naive_q"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=g[xkey], y=g["naive_q"], mode="lines",
            name="naive recompute (n·k)", showlegend=(col == 1),
            legendgroup="naive", line=dict(color=GRAY, width=2, dash="dash"),
            hovertemplate=xkey + "=%{x}<br>n·k=%{y:.0f}<extra>naive</extra>"),
            row=1, col=col)
        # claimed bound shape (k panel only), scaled to eps=0.2
        if sweep == "k":
            ks = sorted(sub["k"].unique())
            shape = [(0.2 ** -2) * k * k * np.log(k) for k in ks]
            fig.add_trace(go.Scatter(
                x=ks, y=shape, mode="lines", name="claimed shape eps⁻²k²log k",
                line=dict(color=AQUA, width=2, dash="dot"),
                hovertemplate="k=%{x}<br>bound shape=%{y:.0f}<extra></extra>"),
                row=1, col=2)
    fig.update_layout(**LAYOUT,
                      title="Claim 2 — per-update oracle queries (steady state)",
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    for ax in ("xaxis", "xaxis2"):
        fig.update_layout({ax: dict(type="log", gridcolor="#e8e7e3")})
    for ax, ttl in (("yaxis", "mean oracle queries / update"), ("yaxis2", "")):
        fig.update_layout({ax: dict(type="log", title=ttl, gridcolor="#e8e7e3")})
    fig.write_html(out_html, include_plotlyjs="cdn")
    print("wrote", out_html)


if __name__ == "__main__":
    fig_ratio(
        [("A1 (audited, submodular)", "claim1_A1_expect.csv"),
         ("A0 ablation (no sampling)", "claim1_A0_ablation.csv"),
         ("control: non-submodular f", "claim1_control_supermod.csv")],
        bound=0.262,
        title="Claim 1 — min expected approximation ratio vs exact OPT",
        out_html="claim1_fig.html", out_csv="claim1_fig_raw.csv")
    fig_scaling("claim2_fig.html", "claim2_fig_raw.csv")
    fig_ratio(
        [("A2 (two-pass + offline combine)", "claim3_A2.csv")],
        bound=0.277,
        title="Claim 3 — min expected approximation ratio vs exact OPT",
        out_html="claim3_fig.html", out_csv="claim3_fig_raw.csv")
