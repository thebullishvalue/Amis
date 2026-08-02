"""
Presentation layer.
=================================================================

Charting rules applied throughout, and why each one is not negotiable here:

* **No dual axes, ever.**  Two measures on two scales invent a correlation
  the data does not contain.  Where two quantities must be compared they are
  either indexed to a common base or drawn as separate panels.
* **Diverging encoding for every oscillator**, blue/red about a neutral
  zero, with one convention held across all three panels: *blue is
  constructive, red is adverse*.  An overvalued Fair Value Oscillator is
  therefore red even though its sign is positive -- the reader learns one
  rule, not three.
* **Sequential (single-hue) encoding for magnitudes** such as market stress;
  **diverging** for anything signed, such as factor contributions.
* **Categorical hues in fixed slot order, never cycled.**  Past six series
  the tail folds into "Other" rather than generating a seventh hue.
* Hairline grid, 2px lines, unified hover crosshair, legend whenever more
  than one series is present.

The palette is the validated reference instance: adjacent-pair CVD deltaE 9.1
(light) / 8.4 (dark), normal-vision 19.6 / 19.3.  Three light-mode slots sit
below 3:1 against the surface, so every chart that uses them ships a legend
and a table view rather than relying on hue alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
LIGHT = {
    "surface": "#fcfcfb", "page": "#f9f9f7",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "neutral": "#f0efec",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "pos": "#2a78d6", "neg": "#e34948",
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
    "critical": "#d03b3b",
}
DARK = {
    "surface": "#1a1a19", "page": "#0d0d0d",
    "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "neutral": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    "pos": "#3987e5", "neg": "#e66767",
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def theme(dark: bool) -> dict:
    return DARK if dark else LIGHT


def _mix(a: str, b: str, t: float) -> str:
    """Linear sRGB blend, used to build the diverging arms from the poles."""
    def rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    ra, rb = rgb(a), rgb(b)
    return "#%02x%02x%02x" % tuple(
        int(round(ra[i] + (rb[i] - ra[i]) * t)) for i in range(3))


def diverging_scale(th: dict) -> list:
    """Two hues plus a neutral midpoint, equal step count per arm."""
    neg, mid, pos = th["neg"], th["neutral"], th["pos"]
    return [
        [0.0, _mix(neg, "#000000", 0.35)], [0.17, neg],
        [0.33, _mix(neg, mid, 0.6)], [0.5, mid],
        [0.67, _mix(pos, mid, 0.6)], [0.83, pos],
        [1.0, _mix(pos, "#000000", 0.35)],
    ]


SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]


def _layout(fig: go.Figure, th: dict, height: int, title: str = "",
            legend: bool = True, ytitle: str = "", xtitle: str = "") -> go.Figure:
    # The top margin has to clear the title *and* the legend row beneath it;
    # axis margins are left to automargin so long tick labels and rotated
    # dates are never clipped.
    top = (56 if legend else 34) if title else (30 if legend else 8)
    fig.update_layout(
        template=None,
        height=height,
        margin=dict(l=8, r=8, t=top, b=8, autoexpand=True),
        paper_bgcolor=th["surface"],
        plot_bgcolor=th["surface"],
        font=dict(family=FONT, size=12, color=th["ink2"]),
        title=dict(text=title, font=dict(size=13, color=th["ink"]), x=0.0,
                   xanchor="left", yref="container", y=1.0, yanchor="top",
                   pad=dict(t=6, l=2)) if title else None,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=th["surface"], font=dict(family=FONT, size=12,
                                                        color=th["ink"]),
                        bordercolor=th["axis"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left",
                    x=0.0, font=dict(size=11, color=th["ink2"]),
                    bgcolor="rgba(0,0,0,0)"),
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=th["axis"],
                     linewidth=1, tickfont=dict(color=th["muted"], size=11),
                     title=dict(text=xtitle, font=dict(size=11, color=th["muted"])),
                     showspikes=True, spikemode="across", spikethickness=1,
                     spikedash="solid", spikecolor=th["axis"], automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=th["grid"], gridwidth=1,
                     zeroline=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(color=th["muted"], size=11), automargin=True,
                     title=dict(text=ytitle, font=dict(size=11, color=th["muted"]),
                                standoff=6))
    return fig


# ---------------------------------------------------------------------------
# 1. Price vs fair value
# ---------------------------------------------------------------------------
def price_vs_fair_value(s: pd.DataFrame, th: dict, label: str,
                        height: int = 420, log_y: bool = True) -> go.Figure:
    d = s.dropna(subset=["fair_value"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["ci_hi"], line=dict(width=0), showlegend=False,
        hoverinfo="skip", name="upper"))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["ci_lo"], line=dict(width=0), fill="tonexty",
        fillcolor=_rgba(th["series"][1], 0.16), name="95% interval",
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["fair_value"], name="Market-implied fair value",
        line=dict(color=th["series"][1], width=2),
        hovertemplate="fair value %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=s.index, y=s["price"], name=f"{label} price",
        line=dict(color=th["series"][0], width=2),
        hovertemplate="price %{y:,.2f}<extra></extra>"))
    _layout(fig, th, height, ytitle="price (log scale)" if log_y else "price")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


# ---------------------------------------------------------------------------
# 2. Oscillators (shared diverging construction)
# ---------------------------------------------------------------------------
def oscillator(s: pd.Series, th: dict, pos_label: str, neg_label: str,
               height: int = 240, title: str = "", ytitle: str = "",
               invert_colour: bool = False, bands=(1.0, 2.0)) -> go.Figure:
    """Signed series drawn as a two-colour fill about zero.

    `invert_colour` flips which sign is drawn in the constructive hue, so the
    Fair Value Oscillator (where *positive* means expensive) reads with the
    same "blue is good" convention as the other two panels.
    """
    v = s.dropna()
    up_c = th["neg"] if invert_colour else th["pos"]
    dn_c = th["pos"] if invert_colour else th["neg"]
    fig = go.Figure()
    pos = v.where(v > 0, 0.0)
    neg = v.where(v < 0, 0.0)
    fig.add_trace(go.Scatter(
        x=v.index, y=pos, name=pos_label, mode="lines",
        line=dict(color=up_c, width=1), fill="tozeroy",
        fillcolor=_rgba(up_c, 0.55), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=v.index, y=neg, name=neg_label, mode="lines",
        line=dict(color=dn_c, width=1), fill="tozeroy",
        fillcolor=_rgba(dn_c, 0.55), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=v.index, y=v, mode="lines", line=dict(width=0), showlegend=False,
        name="value", hovertemplate="%{y:.2f}<extra></extra>"))
    for b in bands:
        for sgn in (1, -1):
            fig.add_hline(y=sgn * b, line=dict(color=th["axis"], width=1))
    fig.add_hline(y=0, line=dict(color=th["ink2"], width=1))
    _layout(fig, th, height, title=title, ytitle=ytitle)
    return fig


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# 3. Regime / stress ribbon
# ---------------------------------------------------------------------------
def regime_ribbon(s: pd.DataFrame, th: dict, height: int = 130) -> go.Figure:
    d = s.dropna(subset=["stress"])
    fig = go.Figure(go.Heatmap(
        x=d.index, y=["market stress"], z=[d["stress"].values],
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        zmin=0, zmax=1, showscale=True,
        colorbar=dict(title=dict(text="percentile", font=dict(size=10,
                                                             color=th["muted"])),
                      thickness=8, len=0.9, outlinewidth=0,
                      tickfont=dict(size=10, color=th["muted"])),
        customdata=np.array([d["regime_label"].values]),
        hovertemplate="stress %{z:.0%}<br>%{customdata}<extra></extra>"))
    _layout(fig, th, height, legend=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=th["muted"]))
    return fig


# ---------------------------------------------------------------------------
# 4. Contribution heatmap (signed -> diverging)
# ---------------------------------------------------------------------------
def contribution_heatmap(df: pd.DataFrame, th: dict, title: str = "",
                         height: int = 380, max_rows: int = 12) -> go.Figure:
    d = df.dropna(how="all")
    if d.empty:
        return _layout(go.Figure(), th, height, title=title, legend=False)
    rank = d.abs().mean().sort_values(ascending=False)
    cols = list(rank.index[:max_rows])
    d = d[cols]
    lim = float(np.nanpercentile(np.abs(d.values), 99)) or 1.0
    fig = go.Figure(go.Heatmap(
        x=d.index, y=cols, z=d.T.values, colorscale=diverging_scale(th),
        zmid=0, zmin=-lim, zmax=lim, showscale=True,
        colorbar=dict(title=dict(text="log-price<br>contribution",
                                 font=dict(size=10, color=th["muted"])),
                      thickness=8, len=0.9, outlinewidth=0,
                      tickfont=dict(size=10, color=th["muted"])),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>%{z:+.3f}<extra></extra>"))
    _layout(fig, th, height, title=title, legend=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=th["muted"]),
                     autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# 5. Multi-series line (fixed slot order, folds past six)
# ---------------------------------------------------------------------------
def multi_line(df: pd.DataFrame, th: dict, height: int = 300, title: str = "",
               ytitle: str = "", max_series: int = 6, fold: bool = True,
               fmt: str = ":.2f") -> go.Figure:
    d = df.dropna(how="all", axis=1).dropna(how="all")
    if d.shape[1] > max_series and fold:
        rank = d.abs().mean().sort_values(ascending=False)
        head = list(rank.index[:max_series - 1])
        tail = [c for c in d.columns if c not in head]
        folded = d[head].copy()
        folded["Other"] = d[tail].sum(axis=1)
        d = folded
    fig = go.Figure()
    for i, c in enumerate(d.columns):
        fig.add_trace(go.Scatter(
            x=d.index, y=d[c], name=str(c),
            line=dict(color=th["series"][i % len(th["series"])], width=2),
            hovertemplate=f"%{{y{fmt}}}<extra>{c}</extra>"))
    _layout(fig, th, height, title=title, ytitle=ytitle,
            legend=d.shape[1] > 1)
    return fig


# ---------------------------------------------------------------------------
# 6. Stacked share (adjacent-pair palette use)
# ---------------------------------------------------------------------------
def stacked_share(df: pd.DataFrame, th: dict, height: int = 220,
                  title: str = "", ytitle: str = "") -> go.Figure:
    d = df.dropna(how="all")
    fig = go.Figure()
    for i, c in enumerate(d.columns):
        fig.add_trace(go.Scatter(
            x=d.index, y=d[c], name=str(c), mode="lines", stackgroup="one",
            line=dict(width=0.5, color=th["surface"]),
            fillcolor=_rgba(th["series"][i % len(th["series"])], 0.85),
            hovertemplate="%{y:.0%}<extra>" + str(c) + "</extra>"))
    _layout(fig, th, height, title=title, ytitle=ytitle)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


# ---------------------------------------------------------------------------
# 7. Signed bar (loadings, importances)
# ---------------------------------------------------------------------------
def signed_bar(s: pd.Series, th: dict, height: int = 340, title: str = "",
               xtitle: str = "") -> go.Figure:
    v = s.dropna().sort_values()
    colours = [th["pos"] if x >= 0 else th["neg"] for x in v.values]
    fig = go.Figure(go.Bar(
        x=v.values, y=[str(i) for i in v.index], orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    _layout(fig, th, height, title=title, legend=False, xtitle=xtitle)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(showgrid=True, gridcolor=th["grid"], zeroline=True,
                     zerolinecolor=th["ink2"], zerolinewidth=1)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=th["ink2"]))
    return fig


def magnitude_bar(s: pd.Series, th: dict, height: int = 340, title: str = "",
                  xtitle: str = "") -> go.Figure:
    """One series, one colour: bar length already encodes magnitude."""
    v = s.dropna().sort_values()
    fig = go.Figure(go.Bar(
        x=v.values, y=[str(i) for i in v.index], orientation="h",
        marker=dict(color=th["series"][0], line=dict(width=0)),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    _layout(fig, th, height, title=title, legend=False, xtitle=xtitle)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(showgrid=True, gridcolor=th["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=th["ink2"]))
    return fig


# ---------------------------------------------------------------------------
# 8. Distribution with the current reading marked
# ---------------------------------------------------------------------------
def distribution(s: pd.Series, th: dict, current: float, height: int = 240,
                 title: str = "", xtitle: str = "") -> go.Figure:
    v = s.dropna()
    fig = go.Figure(go.Histogram(
        x=v, nbinsx=70, marker=dict(color=_rgba(th["series"][0], 0.75),
                                    line=dict(width=0)),
        hovertemplate="%{x:.2f}: %{y} sessions<extra></extra>", name="history"))
    if np.isfinite(current):
        fig.add_vline(x=current, line=dict(color=th["ink"], width=2))
        fig.add_annotation(x=current, y=1, yref="paper", yanchor="bottom",
                           text=f"today {current:+.2f}", showarrow=False,
                           font=dict(size=11, color=th["ink"]))
    _layout(fig, th, height, title=title, legend=False, xtitle=xtitle,
            ytitle="sessions")
    fig.update_layout(hovermode="x")
    return fig


# ---------------------------------------------------------------------------
# 9. Decision annotations on the price path
# ---------------------------------------------------------------------------
def decision_annotations(s: pd.DataFrame, th: dict, height: int = 420,
                         top_n: int = 40) -> go.Figure:
    d = s.dropna(subset=["pdo", "price"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["price"], name="price",
        line=dict(color=th["muted"], width=1.5),
        hovertemplate="price %{y:,.2f}<extra></extra>"))
    if len(d):
        strong = d.reindex(d["pdo"].abs().sort_values(ascending=False).index[:top_n])
        strong = strong.sort_index()
        for sign, colour, name in ((1, th["pos"], "constructive extreme"),
                                   (-1, th["neg"], "adverse extreme")):
            sel = strong[np.sign(strong["pdo"]) == sign]
            if not len(sel):
                continue
            fig.add_trace(go.Scatter(
                x=sel.index, y=sel["price"], mode="markers", name=name,
                marker=dict(color=colour, size=9, line=dict(color=th["surface"],
                                                            width=2)),
                customdata=np.stack([sel["pdo"], sel["conviction"]], axis=-1),
                hovertemplate=("%{x|%Y-%m-%d}<br>PDO %{customdata[0]:+.2f}"
                               "<br>conviction %{customdata[1]:.0%}<extra></extra>")))
    _layout(fig, th, height, ytitle="price (log scale)")
    fig.update_yaxes(type="log")
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# 10. Cumulative walk-forward curves
# ---------------------------------------------------------------------------
def walk_forward_curves(curves: dict, th: dict, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=c.index, y=c.values, name=name,
            line=dict(color=th["series"][i % len(th["series"])], width=2),
            hovertemplate="%{y:.1f}<extra>" + name + "</extra>"))
    _layout(fig, th, height,
            ytitle="cumulative standardised outcome",
            title="21-session horizon, position proportional to signal")
    fig.add_hline(y=0, line=dict(color=th["axis"], width=1))
    return fig
