"""Design system for the dashboard: validated palette, Plotly theme, UX components and cached loaders.

Colour rules (checked with the data-viz palette validator, dark surface #121722):
* categorical series come from one validated 8-hue set in a fixed order, never cycled; colour follows the entity
* the BG / "our model" series is the yellow slot; amber chrome (ACCENT_UI) is used only outside charts
* text never wears a data colour: values and labels use the three ink tokens (TEXT, TEXT_2, MUTED)
* every chart gets a table view and a screen-reader description (T.chart does both automatically)
"""
from __future__ import annotations

import contextlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOCAL_TZ = "Europe/Sofia"

# --- surfaces & ink (contrast on the card surface: TEXT 15.0, TEXT_2 7.8, MUTED 5.8) ------------------
BG = "#0B0E14"          # page
SURFACE = "#121722"     # cards
SURFACE_2 = "#181F2C"   # raised / hover
BORDER = "#222B3A"
GRID = "#1F2735"
TEXT = "#E8EBF0"
TEXT_2 = "#A4ACBA"
MUTED = "#8A93A6"       # smallest text: >= 5.8:1 (was 3.9:1)
MUTED_DIM = "#566074"   # non-text only (hairlines, inactive marks)

# --- categorical slots: validated dark set, fixed order -------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#3987e5", "#d95926", "#199e70", "#c98500",
                                                            "#d55181", "#008300", "#9085e9", "#e66767")
ACCENT = YELLOW                 # BG / "our model" in charts
ACCENT_UI = "#F5A524"           # chrome only (brand mark, focus ring, active nav) - never a data mark
ACCENT_SOFT = "rgba(201,133,0,0.14)"
NEUTRAL = "#5A6476"             # baselines and context series ("naive")
UP, DOWN = RED, BLUE            # diverging poles: above / below
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
RISK = {"low": STATUS["good"], "medium": STATUS["warning"], "high": STATUS["critical"]}
# validated all-pairs (scatter / small multiples): yellow, blue, magenta, green
ZONE = {"BG": YELLOW, "RO": BLUE, "GR": MAGENTA, "HU": GREEN}
MODEL = {"actual": TEXT, "naive_d1": NEUTRAL, "naive_w1": "#434C5C", "ridge": BLUE, "lightgbm": YELLOW,
         "lightgbm_level": VIOLET, "lightgbm_blend": AQUA, "mlp": MAGENTA, "ensemble": GREEN,
         "lightgbm_nofuel": "#8B6B3A", "lightgbm_fuels": "#B08A4A", "lightgbm_nocapacity": "#7A6A4A", "lightgbm_capacity": "#9C8A5A"}
MODEL_LABEL = {"naive_d1": "Naive · yesterday", "naive_w1": "Naive · last week", "ridge": "Ridge",
               "lightgbm": "LightGBM", "lightgbm_level": "LightGBM · level-adj.", "lightgbm_blend": "Blend",
               "lightgbm_nofuel": "LightGBM · no fuel prices", "lightgbm_fuels": "LightGBM · + fuel prices",
               "lightgbm_nocapacity": "LightGBM · no capacity (NTC)", "lightgbm_capacity": "LightGBM · + capacity (NTC)",
               "mlp": "Neural net (MLP)", "ensemble": "Ensemble", "actual": "Actual"}
# generation technologies folded to <= 8 groups, stacked bottom -> top in an order that passes the adjacent-pair CVD checks
GEN_ORDER = ["Nuclear", "Lignite", "Hydro", "Gas", "Solar", "Wind", "Other"]
GEN_GROUP_COLOR = {"Nuclear": VIOLET, "Lignite": ORANGE, "Hydro": BLUE, "Gas": MAGENTA, "Solar": YELLOW, "Wind": AQUA, "Other": NEUTRAL}
GEN_LABEL = {"nuclear": "Nuclear", "fossil_brown_coal_lignite": "Lignite", "fossil_gas": "Gas", "fossil_hard_coal": "Hard coal",
             "hydro_run_of_river_and_pondage": "Hydro run-of-river", "hydro_water_reservoir": "Hydro reservoir",
             "hydro_pumped_storage": "Pumped storage", "solar": "Solar", "wind_onshore": "Wind", "biomass": "Biomass",
             "waste": "Waste", "energy_storage": "Battery storage", "fossil_oil": "Oil", "other": "Other"}


def gen_group(tech: str) -> str:
    if tech.startswith("hydro"):
        return "Hydro"
    return {"nuclear": "Nuclear", "fossil_brown_coal_lignite": "Lignite", "fossil_gas": "Gas", "solar": "Solar",
            "wind_onshore": "Wind"}.get(tech, "Other")


# sequential (one hue, receding to the surface at zero) and diverging (blue - gray - red, equal arms)
SEQ = [[0, "#151B26"], [0.18, "#2A2110"], [0.4, "#5F4510"], [0.62, "#A06C0F"], [0.82, "#F0B232"], [1, "#FFE6B0"]]
DIV = [[0, "#1C5CAB"], [0.25, "#5598E7"], [0.5, "#383835"], [0.75, "#E66767"], [1, "#B83A3A"]]
COUPLED_TOL = 1.0   # |spread| below this counts as "coupled": since mid-2026 coupled zones differ by cents (EUR/MWh)
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# --- Plotly theme -------------------------------------------------------------------------------------------------
_axis = dict(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, ticks="", tickfont=dict(size=12, color=MUTED),
             title=dict(font=dict(size=12, color=TEXT_2), standoff=8), automargin=True)
pio.templates["ema"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT_2, size=12),
    xaxis={**_axis, "showgrid": False, "showspikes": True, "spikemode": "across", "spikethickness": 1,
           "spikecolor": MUTED_DIM, "spikedash": "solid", "spikesnap": "cursor"},
    yaxis={**_axis, "showgrid": True},
    colorway=[YELLOW, BLUE, MAGENTA, GREEN, VIOLET, AQUA],
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                font=dict(size=12, color=TEXT_2), itemsizing="constant", title=None),
    margin=dict(l=4, r=8, t=24, b=4),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(family=FONT, size=13, color=TEXT)),
    hovermode="x unified", hoverdistance=40, bargap=0.2, bargroupgap=0.08,
    coloraxis=dict(colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=11, color=MUTED))),
))
pio.templates.default = "plotly_dark+ema"
PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False, "responsive": True}


def range_buttons(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(rangeselector=dict(
        buttons=[dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"),
                 dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all", label="All")],
        bgcolor=SURFACE_2, activecolor=rgba(ACCENT, 0.5), bordercolor=BORDER, borderwidth=1,
        font=dict(size=12, color=TEXT_2), x=1, xanchor="right", y=1.02, yanchor="bottom"))
    return fig


def direct_label(fig: go.Figure, x, y, text: str, color: str, yshift: int = 0, xshift: int = 8) -> None:
    """End-of-line label: coloured dot (identity) + text in an ink token (text never wears the data colour)."""
    fig.add_annotation(x=x, y=y, text=f"<span style='color:{color}'>●</span> {html.escape(text)}", showarrow=False,
                       xanchor="left", xshift=xshift, yshift=yshift, font=dict(size=12, color=TEXT_2))


def annotate(fig: go.Figure, x, y, text: str, ay: int = -34, ax: int = 0) -> None:
    """A single labelled event/extreme with a thin leader line (use sparingly: the story, not every point)."""
    fig.add_annotation(x=x, y=y, text=html.escape(text), showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=MUTED_DIM,
                       ax=ax, ay=ay, font=dict(size=12, color=TEXT), bgcolor=SURFACE_2, bordercolor=BORDER, borderpad=4)


# --- table twin and description for every chart ----------------------------------------------------------------------
def fig_to_df(fig: go.Figure, max_cells: int = 6000) -> pd.DataFrame | None:
    """Best-effort tabular equivalent of a figure (the WCAG-clean twin of the chart). None if it has no sensible table."""
    series, hm = [], None
    for t in fig.data:
        typ = t.type
        name = (getattr(t, "name", None) or "").strip()
        if typ in ("scatter", "scattergl", "bar", "waterfall") and getattr(t, "x", None) is not None and getattr(t, "y", None) is not None:
            if not name or getattr(t, "hoverinfo", None) == "skip" or getattr(t, "showlegend", True) is False:
                continue
            horizontal = getattr(t, "orientation", None) == "h"
            cat, val = (t.y, t.x) if horizontal else (t.x, t.y)
            if cat is None or val is None or len(cat) != len(val):
                continue
            s = pd.Series(list(val), index=list(cat), name=name)
            series.append(s[~s.index.duplicated(keep="first")])
        elif typ == "heatmap" and hm is None and t.z is not None:
            z = np.asarray(t.z, dtype=float)
            if z.size <= max_cells:
                hm = pd.DataFrame(z, index=[str(v) for v in t.y], columns=[str(v)[:10] for v in t.x])
        elif typ == "pie" and t.labels is not None:
            series.append(pd.Series(list(t.values), index=list(t.labels), name=name or "value"))
    if hm is not None:
        return hm
    if not series:
        return None
    df = pd.concat(series, axis=1)
    if df.size > max_cells:
        return None
    if isinstance(df.index, pd.DatetimeIndex) or (len(df) and isinstance(df.index[0], (pd.Timestamp, np.datetime64))):
        idx = pd.to_datetime(df.index)
        df.index = idx.strftime("%Y-%m-%d %H:%M" if (idx.hour != 0).any() else "%Y-%m-%d")
    return df


_CURRENT = {"title": "", "subtitle": "", "n": 0}


def chart(fig: go.Figure, height: int = 360, legend: bool = True, key: str | None = None, table: bool = True,
          desc: str = "") -> None:
    """Render a themed Plotly chart + a screen-reader description + a collapsed table view with CSV download."""
    fig.update_layout(height=height, showlegend=legend)
    _CURRENT["n"] += 1
    title, sub = _CURRENT["title"], _CURRENT["subtitle"]
    label = " ".join(x for x in (title, sub, desc) if x)
    if label:
        st.markdown(f"<div class='sr-only' role='note'>Chart: {html.escape(label)}</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, config=PLOT_CONFIG, width="stretch", key=key or f"chart_{_CURRENT['n']}")
    if table:
        df = fig_to_df(fig)
        if df is not None and len(df):
            with st.expander("Table view", icon=":material/table:"):
                shown = df.head(120)
                num = {c: "{:,.2f}" for c in shown.columns if pd.api.types.is_numeric_dtype(shown[c])}
                shown = shown.copy()
                shown.columns = [str(c) for c in shown.columns]
                table_html(shown, num)
                if len(df) > 120:
                    st.caption(f"Showing the first 120 of {len(df):,} rows. Download for the full data.")
                st.download_button("Download CSV", df.to_csv().encode("utf-8"), file_name=f"{_slug(title or 'chart')}.csv",
                                   mime="text/csv", key=f"dl_{_CURRENT['n']}", icon=":material/download:")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-")[:48] or "chart"


# --- page chrome ---------------------------------------------------------------------------------------------------------
def setup(title: str, icon: str = "⚡") -> None:
    st.set_page_config(page_title=f"{title} · BG Power", page_icon=icon, layout="wide")
    css = (ROOT / "app" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    _CURRENT.update(title="", subtitle="", n=0)
    with st.sidebar:
        m = load_market()
        through = f"Data through {m.index.max():%d %b %Y}" if m is not None else "No data loaded"
        st.markdown(f"<div class='brand'><div class='brand-mark' aria-hidden='true'>⚡</div><div><div class='brand-name'>BG Power</div>"
                    f"<div class='brand-sub'>Day-ahead analytics</div></div></div>"
                    f"<div class='side-meta' role='status'><span class='dot' aria-hidden='true'></span>{through}<br>"
                    f"ENTSO-E · Open-Meteo<br><span class='side-hint'>Every chart has a <b>Table view</b> and CSV download.</span></div>",
                    unsafe_allow_html=True)


def header(eyebrow: str, title: str, subtitle: str = "", badges: list[str] | None = None, headline: str = "") -> None:
    """Page header. `headline` is the one-sentence answer for the page (computed from data), shown first."""
    b = "".join(f"<span class='badge'>{html.escape(x)}</span>" for x in badges or [])
    hl = f"<div class='headline' role='note'><span class='hl-tag'>Headline</span><span>{headline}</span></div>" if headline else ""
    st.markdown(f"<header class='page-head'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1>"
                f"<p class='lede'>{subtitle}</p>{hl}<div class='badges'>{b}</div></header>", unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f"<section class='section'><h2>{title}</h2>" + (f"<p>{subtitle}</p>" if subtitle else "") + "</section>",
                unsafe_allow_html=True)


_card_n = 0


@contextlib.contextmanager
def card(title: str = "", subtitle: str = "", right: str = "", help: str = ""):
    """Bordered surface. Title should state the finding or the question; subtitle says how to read it. `help` adds an (i) tip."""
    global _card_n
    _card_n += 1
    _CURRENT["title"], _CURRENT["subtitle"] = html.unescape(title), html.unescape(subtitle)
    with st.container(key=f"card_{_card_n}"):
        if title:
            tip = (f"<span class='info' tabindex='0' role='img' aria-label='{html.escape(help, quote=True)}' "
                   f"data-tip='{html.escape(help, quote=True)}'>i</span>") if help else ""
            st.markdown(f"<div class='card-head'><div><div class='card-title'>{title}{tip}</div>"
                        + (f"<div class='card-sub'>{subtitle}</div>" if subtitle else "")
                        + f"</div><div class='card-right'>{right}</div></div>", unsafe_allow_html=True)
        yield


@contextlib.contextmanager
def controls(label: str = "Filters"):
    """One filter row above the content it scopes (never inside a chart card)."""
    global _card_n
    _card_n += 1
    with st.container(key=f"controls_{_card_n}"):
        st.markdown(f"<div class='controls-label'>{label}</div>", unsafe_allow_html=True)
        yield


def sparkline(values, color: str = ACCENT, w: int = 120, h: int = 34) -> str:
    v = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    span = hi - lo or 1
    xs = np.linspace(1, w - 1, len(v))
    ys = h - 4 - (v - lo) / span * (h - 8)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"1,{h} " + pts + f" {w - 1},{h}"
    gid = f"g{abs(hash(pts)) % 10**8}"
    return (f"<svg class='spark' viewBox='0 0 {w} {h}' preserveAspectRatio='none' role='img' aria-label='trend, last value highlighted'>"
            f"<defs><linearGradient id='{gid}' x1='0' y1='0' x2='0' y2='1'>"
            f"<stop offset='0%' stop-color='{color}' stop-opacity='0.30'/><stop offset='100%' stop-color='{color}' stop-opacity='0'/></linearGradient></defs>"
            f"<polygon points='{area}' fill='url(#{gid})'/>"
            f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/>"
            f"<circle cx='{xs[-1]:.1f}' cy='{ys[-1]:.1f}' r='4' fill='{color}' stroke='{SURFACE}' stroke-width='2'/></svg>")


def delta_chip(delta: float | None, unit: str = "%", good_when_up: bool | None = None, digits: int = 1) -> str:
    """Signed change with an arrow AND a word for assistive tech (colour is never the only cue)."""
    if delta is None or not np.isfinite(delta):
        return ""
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "■"
    word = "up" if delta > 0 else "down" if delta < 0 else "flat"
    cls = "flat"
    if good_when_up is not None and delta != 0:
        cls = "pos" if (delta > 0) == good_when_up else "neg"
    elif delta != 0:
        cls = "up" if delta > 0 else "down"
    return f"<span class='chip {cls}'><span aria-hidden='true'>{arrow}</span><span class='sr-only'>{word} </span>{abs(delta):,.{digits}f}{unit}</span>"


def kpi(col, label: str, value: str, sub: str = "", accent: bool = False, spark=None,
        spark_color: str | None = None, delta: str = "", help: str = "") -> None:
    sp = sparkline(spark, spark_color or (ACCENT if accent else TEXT_2)) if spark is not None else ""
    tip = f" title='{html.escape(help, quote=True)}'" if help else ""
    col.markdown(f"<div class='kpi{' accent' if accent else ''}'{tip}><div class='label'>{label}</div>"
                 f"<div class='kpi-val'><span class='value'>{value}</span>{delta}</div>"
                 f"<div class='kpi-bottom'><span class='sub'>{sub}</span>{sp}</div></div>", unsafe_allow_html=True)


def takeaway(text: str, kind: str = "insight") -> None:
    icon = {"insight": "◆", "action": "→", "warning": "!"}[kind]
    label = {"insight": "Insight", "action": "Action", "warning": "Caution"}[kind]
    st.markdown(f"<div class='takeaway {kind}' role='note'><span class='ti' aria-hidden='true'>{icon}</span>"
                f"<div><span class='sr-only'>{label}: </span>{text}</div></div>", unsafe_allow_html=True)


def legend_html(items: list[tuple[str, str]], kind: str = "line") -> str:
    return "".join(f"<span class='lg'><i class='{kind}' style='background:{c}'></i>{html.escape(n)}</span>" for n, c in items)


def table_html(df: pd.DataFrame, fmt: dict | None = None, bars: dict | None = None, highlight: str | None = None,
               badges: dict | None = None) -> None:
    fmt, bars, badges = fmt or {}, bars or {}, badges or {}
    cols = list(df.columns)
    head = f"<th scope='col'>{html.escape(str(df.index.name or ''))}</th>" + "".join(f"<th scope='col'>{html.escape(str(c))}</th>" for c in cols)
    rng = {}
    for c in bars:
        v = pd.to_numeric(df[c], errors="coerce")
        rng[c] = (np.nanmin(v), np.nanmax(v))
    rows = []
    for idx, r in df.iterrows():
        tds = [f"<th scope='row' class='rowh'>{html.escape(str(idx))}</th>"]
        for c in cols:
            v = r[c]
            if c in badges:
                txt, kind = badges[c](v)
                tds.append(f"<td><span class='tbadge {kind}'>{txt}</span></td>")
                continue
            isnum = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            s = "—" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else (fmt[c].format(v) if c in fmt else html.escape(str(v)))
            if c in bars and isnum and np.isfinite(v):
                lo, hi = rng[c]
                pct = 100 * (v - min(lo, 0)) / ((hi - min(lo, 0)) or 1)
                tds.append(f"<td class='num'><div class='dbar'><span style='width:{pct:.0f}%;background:{bars[c]}'></span><em>{s}</em></div></td>")
            else:
                tds.append(f"<td class='{'num' if isnum or c in fmt else ''}'>{s}</td>")
        cls = " class='hl'" if highlight is not None and str(idx) == str(highlight) else ""
        rows.append(f"<tr{cls}>{''.join(tds)}</tr>")
    st.markdown(f"<div class='tbl-wrap' role='region' tabindex='0' aria-label='data table'><table class='tbl'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


table = table_html


def missing(name: str) -> None:
    st.markdown(f"<div class='empty' role='status'><div class='empty-icon' aria-hidden='true'>◌</div><div><b>{name}</b> is not available yet.<br>"
                f"<span>Run <code>python -m src.run_all --plus</code> to generate the reports.</span></div></div>", unsafe_allow_html=True)


def download(df: pd.DataFrame, name: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=True).encode("utf-8"), file_name=f"{name}.csv", mime="text/csv",
                       key=key or f"dlb_{name}", icon=":material/download:")


# --- loaders -----------------------------------------------------------------
# Every cached loader takes the file's mtime as an argument, so a pipeline re-run
# invalidates the cache without restarting the dashboard.
def _mtime(name: str) -> float | None:
    p = REPORTS / name
    return p.stat().st_mtime if p.exists() else None


@st.cache_data(show_spinner=False)
def _market(mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(REPORTS / "market_hourly.parquet")
    df.index = pd.DatetimeIndex(df.index).tz_convert(LOCAL_TZ)
    return df


def load_market() -> pd.DataFrame | None:
    mt = _mtime("market_hourly.parquet")
    return None if mt is None else _market(mt)


@st.cache_data(show_spinner=False)
def _csv(name: str, mtime: float, kw: tuple) -> pd.DataFrame | None:
    try:
        return pd.read_csv(REPORTS / name, **dict(kw))
    except Exception:
        return None


def load_csv(name: str, **kw) -> pd.DataFrame | None:
    mt = _mtime(name)
    if mt is None:
        return None
    df = _csv(name, mt, tuple(sorted(kw.items())))
    return None if df is None else df.copy()


@st.cache_data(show_spinner=False)
def _predictions(mtime: float) -> pd.DataFrame | None:
    df = _csv("predictions.csv", mtime, ())
    if df is None:
        return None
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["time_local"] = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    df["delivery_day"] = pd.to_datetime(df["delivery_day"])
    return df.set_index("time_local")


def load_predictions() -> pd.DataFrame | None:
    mt = _mtime("predictions.csv")
    return None if mt is None else _predictions(mt)


def load_json(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def gen_color(name: str) -> str:
    return GEN_GROUP_COLOR.get(gen_group(name), NEUTRAL)


def gen_label(name: str) -> str:
    return GEN_LABEL.get(name, name.replace("_", " ").capitalize())


def fmt_eur(x: float, d: int = 1) -> str:
    return f"€{x:,.{d}f}"
