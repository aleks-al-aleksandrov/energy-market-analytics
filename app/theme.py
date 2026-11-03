"""Design system for the dashboard: validated palettes (dark + light), Plotly themes, UX components and cached loaders.

Colour rules (checked with the data-viz palette validator; dark surface #121722, light surface #FDFCFA):
* categorical series come from one validated 8-hue set in a fixed order, never cycled; colour follows the entity
* the BG / "our model" series is the yellow slot; amber chrome (ACCENT_UI) is used only outside charts
* text never wears a data colour: values and labels use the three ink tokens (TEXT, TEXT_2, MUTED)
* every chart gets a table view and a screen-reader description (T.chart does both automatically)

Dark and light are two selected palettes, not an automatic flip. Colours are resolved per session at call time
(`T.TEXT` -> module __getattr__ -> the palette of the current session's mode), so two viewers on different themes never
see each other's colours.
"""
from __future__ import annotations

import contextlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from i18n import tr

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOCAL_TZ = "Europe/Sofia"

# --- palettes: surfaces & ink, categorical slots (fixed order), sequential + diverging ramps --------------------------
_DARK = dict(
    BG="#0B0E14", SURFACE="#121722", SURFACE_2="#181F2C", BORDER="#222B3A", GRID="#1F2735",
    TEXT="#E8EBF0", TEXT_2="#A4ACBA", MUTED="#8A93A6", MUTED_DIM="#566074",       # contrast on surface: 15.0 / 7.8 / 5.8
    BLUE="#3987e5", ORANGE="#d95926", AQUA="#199e70", YELLOW="#c98500", MAGENTA="#d55181", GREEN="#008300",
    VIOLET="#9085e9", RED="#e66767",
    ACCENT_UI="#F5A524", NEUTRAL="#5A6476", NAIVE_W1="#434C5C",
    SEQ=[[0, "#151B26"], [0.18, "#2A2110"], [0.4, "#5F4510"], [0.62, "#A06C0F"], [0.82, "#F0B232"], [1, "#FFE6B0"]],
    DIV=[[0, "#1C5CAB"], [0.25, "#5598E7"], [0.5, "#383835"], [0.75, "#E66767"], [1, "#B83A3A"]],
    FUEL=["#8B6B3A", "#B08A4A", "#7A6A4A", "#9C8A5A"],
)
_LIGHT = dict(
    BG="#F3F1EB", SURFACE="#FDFCFA", SURFACE_2="#F0EDE6", BORDER="#D9D4C8", GRID="#E4E0D6",
    TEXT="#14161A", TEXT_2="#474B54", MUTED="#666B76", MUTED_DIM="#A4A7AF",       # contrast on surface: 17.7 / 8.5 / 5.2
    BLUE="#2a78d6", ORANGE="#eb6834", AQUA="#1baf7a", YELLOW="#eda100", MAGENTA="#e87ba4", GREEN="#008300",
    VIOLET="#6250d6", RED="#e34948",
    ACCENT_UI="#A64A06", NEUTRAL="#7B8190", NAIVE_W1="#B3B7C0",
    SEQ=[[0, "#F7F2E6"], [0.18, "#F6E2B3"], [0.4, "#F0C268"], [0.62, "#D99A1B"], [0.82, "#9A5B0A"], [1, "#563004"]],
    DIV=[[0, "#1C5CAB"], [0.25, "#86B6EF"], [0.5, "#F0EFEC"], [0.75, "#F1A1A0"], [1, "#B83A3A"]],
    FUEL=["#8B6B3A", "#B08A4A", "#7A6A4A", "#9C8A5A"],
)
PALETTES: dict[str, dict] = {}
for _m, _p in (("dark", _DARK), ("light", _LIGHT)):
    _p = dict(_p)
    _p["ACCENT"] = _p["YELLOW"]                       # BG / "our model" in charts
    _p["ACCENT_SOFT"] = "rgba(201,133,0,0.14)" if _m == "dark" else "rgba(237,161,0,0.16)"
    _p["UP"], _p["DOWN"] = _p["RED"], _p["BLUE"]       # diverging poles: above / below
    _p["STATUS"] = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
    _p["RISK"] = {"low": _p["STATUS"]["good"], "medium": _p["STATUS"]["warning"], "high": _p["STATUS"]["critical"]}
    # validated all-pairs (scatter / small multiples): yellow, blue, magenta, green
    _p["ZONE"] = {"BG": _p["YELLOW"], "RO": _p["BLUE"], "GR": _p["MAGENTA"], "HU": _p["GREEN"]}
    _p["MODEL"] = {"actual": _p["TEXT"], "naive_d1": _p["NEUTRAL"], "naive_w1": _p["NAIVE_W1"], "ridge": _p["BLUE"],
                   "lightgbm": _p["YELLOW"], "lightgbm_level": _p["VIOLET"], "lightgbm_blend": _p["AQUA"], "mlp": _p["MAGENTA"],
                   "ensemble": _p["GREEN"], "lightgbm_nofuel": _p["FUEL"][0], "lightgbm_fuels": _p["FUEL"][1],
                   "lightgbm_nocapacity": _p["FUEL"][2], "lightgbm_capacity": _p["FUEL"][3]}
    _p["GEN_GROUP_COLOR"] = {"Nuclear": _p["VIOLET"], "Lignite": _p["ORANGE"], "Hydro": _p["BLUE"], "Gas": _p["MAGENTA"],
                             "Solar": _p["YELLOW"], "Wind": _p["AQUA"], "Other": _p["NEUTRAL"]}
    PALETTES[_m] = _p
del _m, _p

MODES = ("dark", "light")


def mode() -> str:
    """The current session's theme mode ('dark' | 'light'). Falls back to dark outside a running session."""
    try:
        m = st.session_state.get("_mode", "dark")
    except Exception:
        m = "dark"
    return m if m in PALETTES else "dark"


def __getattr__(name: str):
    """Palette names (T.TEXT, T.MODEL, T.SEQ, ...) resolve per session, at call time."""
    p = PALETTES[mode()]
    if name in p:
        return p[name]
    raise AttributeError(f"module 'theme' has no attribute {name!r}")


class _Live:
    def __getattr__(self, name: str):
        return PALETTES[mode()][name]


P = _Live()      # inside this module: P.TEXT instead of the (non-existent) global TEXT

MODEL_LABEL = {"naive_d1": "Naive · yesterday", "naive_w1": "Naive · last week", "ridge": "Ridge",
               "lightgbm": "LightGBM", "lightgbm_level": "LightGBM · level-adj.", "lightgbm_blend": "Blend",
               "lightgbm_nofuel": "LightGBM · no fuel prices", "lightgbm_fuels": "LightGBM · + fuel prices",
               "lightgbm_nocapacity": "LightGBM · no capacity (NTC)", "lightgbm_capacity": "LightGBM · + capacity (NTC)",
               "mlp": "Neural net (MLP)", "ensemble": "Ensemble", "actual": "Actual"}
# generation technologies folded to <= 8 groups, stacked bottom -> top in an order that passes the adjacent-pair CVD checks
GEN_ORDER = ["Nuclear", "Lignite", "Hydro", "Gas", "Solar", "Wind", "Other"]
GEN_LABEL = {"nuclear": "Nuclear", "fossil_brown_coal_lignite": "Lignite", "fossil_gas": "Gas", "fossil_hard_coal": "Hard coal",
             "hydro_run_of_river_and_pondage": "Hydro run-of-river", "hydro_water_reservoir": "Hydro reservoir",
             "hydro_pumped_storage": "Pumped storage", "solar": "Solar", "wind_onshore": "Wind", "biomass": "Biomass",
             "waste": "Waste", "energy_storage": "Battery storage", "fossil_oil": "Oil", "other": "Other"}


def gen_group(tech: str) -> str:
    if tech.startswith("hydro"):
        return "Hydro"
    return {"nuclear": "Nuclear", "fossil_brown_coal_lignite": "Lignite", "fossil_gas": "Gas", "solar": "Solar",
            "wind_onshore": "Wind"}.get(tech, "Other")


COUPLED_TOL = 1.0   # |spread| below this counts as "coupled": since mid-2026 coupled zones differ by cents (EUR/MWh)
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# --- Plotly themes (one per mode, chosen per chart) ----------------------------------------------------------------------
def _build_template(m: str) -> go.layout.Template:
    p = PALETTES[m]
    axis = dict(gridcolor=p["GRID"], gridwidth=1, zeroline=False, showline=False, ticks="", tickfont=dict(size=12, color=p["MUTED"]),
                title=dict(font=dict(size=12, color=p["TEXT_2"]), standoff=8), automargin=True)
    return go.layout.Template(layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=p["TEXT_2"], size=12),
        xaxis={**axis, "showgrid": False, "showspikes": True, "spikemode": "across", "spikethickness": 1,
               "spikecolor": p["MUTED_DIM"], "spikedash": "solid", "spikesnap": "cursor"},
        yaxis={**axis, "showgrid": True},
        colorway=[p["YELLOW"], p["BLUE"], p["MAGENTA"], p["GREEN"], p["VIOLET"], p["AQUA"]],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    font=dict(size=12, color=p["TEXT_2"]), itemsizing="constant", title=None),
        margin=dict(l=4, r=8, t=24, b=4),
        hoverlabel=dict(bgcolor=p["SURFACE_2"], bordercolor=p["BORDER"], font=dict(family=FONT, size=13, color=p["TEXT"])),
        hovermode="x unified", hoverdistance=40, bargap=0.2, bargroupgap=0.08,
        coloraxis=dict(colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=11, color=p["MUTED"]))),
    ))


for _m in MODES:
    pio.templates[f"ema_{_m}"] = _build_template(_m)
pio.templates.default = "plotly_dark+ema_dark"
del _m
PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False, "responsive": True}


def template() -> str:
    m = mode()
    return f"{'plotly_dark' if m == 'dark' else 'plotly_white'}+ema_{m}"


def range_buttons(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(rangeselector=dict(
        buttons=[dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"),
                 dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all", label="All")],
        bgcolor=P.SURFACE_2, activecolor=rgba(P.ACCENT, 0.5), bordercolor=P.BORDER, borderwidth=1,
        font=dict(size=12, color=P.TEXT_2), x=1, xanchor="right", y=1.02, yanchor="bottom"))
    return fig


def direct_label(fig: go.Figure, x, y, text: str, color: str, yshift: int = 0, xshift: int = 8) -> None:
    """End-of-line label: coloured dot (identity) + text in an ink token (text never wears the data colour)."""
    fig.add_annotation(x=x, y=y, text=f"<span style='color:{color}'>●</span> {html.escape(text)}", showarrow=False,
                       xanchor="left", xshift=xshift, yshift=yshift, font=dict(size=12, color=P.TEXT_2))


def annotate(fig: go.Figure, x, y, text: str, ay: int = -34, ax: int = 0) -> None:
    """A single labelled event/extreme with a thin leader line (use sparingly: the story, not every point)."""
    fig.add_annotation(x=x, y=y, text=html.escape(text), showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=P.MUTED_DIM,
                       ax=ax, ay=ay, font=dict(size=12, color=P.TEXT), bgcolor=P.SURFACE_2, bordercolor=P.BORDER, borderpad=4)


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
        elif typ == "barpolar" and getattr(t, "theta", None) is not None and getattr(t, "r", None) is not None:
            s_ = pd.Series(list(t.r), index=[str(v) for v in t.theta], name=name or "value")
            series.append(s_[~s_.index.duplicated(keep="first")])
        elif typ == "pie" and t.labels is not None:
            series.append(pd.Series(list(t.values), index=list(t.labels), name=name or "value"))
    if hm is not None:
        return hm
    if not series:
        return None
    df = pd.concat(series, axis=1, sort=False)
    if df.size > max_cells:
        return None
    if isinstance(df.index, pd.DatetimeIndex) or (len(df) and isinstance(df.index[0], (pd.Timestamp, np.datetime64))):
        idx = pd.to_datetime(df.index)
        df.index = idx.strftime("%Y-%m-%d %H:%M" if (idx.hour != 0).any() else "%Y-%m-%d")
    return df


_CURRENT = {"title": "", "subtitle": "", "n": 0}


def _localize_fig(fig: go.Figure) -> None:
    """Translate the words a viewer reads in a figure (axis titles, series names, annotations, hover text)."""
    from i18n import lang
    if lang() == "en":
        return
    for ax in (fig.layout.xaxis, fig.layout.yaxis):
        if ax.title and ax.title.text:
            ax.title.text = tr(ax.title.text)
    for k in list(fig.layout):
        if k.startswith(("xaxis", "yaxis")) and fig.layout[k].title and fig.layout[k].title.text:
            fig.layout[k].title.text = tr(fig.layout[k].title.text)
    for t in fig.data:
        if getattr(t, "name", None):
            t.name = tr(t.name)
        if getattr(t, "hovertemplate", None):
            t.hovertemplate = tr(t.hovertemplate)
        if t.type == "heatmap" and getattr(t, "colorbar", None) and t.colorbar.title and t.colorbar.title.text:
            t.colorbar.title.text = tr(t.colorbar.title.text)
    for a in fig.layout.annotations:
        if a.text and "<span" not in a.text:
            a.text = tr(a.text)
    for um in fig.layout.updatemenus:
        for b in um.buttons:
            b.label = tr(b.label)


def chart(fig: go.Figure, height: int = 360, legend: bool = True, key: str | None = None, table: bool = True,
          desc: str = "") -> None:
    """Render a themed Plotly chart + a screen-reader description + a collapsed table view with CSV download."""
    _localize_fig(fig)
    fig.update_layout(height=height, showlegend=legend, template=template(), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=P.TEXT_2, size=12))
    _CURRENT["n"] += 1
    title, sub = _CURRENT["title"], _CURRENT["subtitle"]
    label = " ".join(tr(x) for x in (title, sub, desc) if x)
    if label:
        st.markdown(f"<div class='sr-only' role='note'>{tr('Chart')}: {html.escape(label)}</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, config=PLOT_CONFIG, width="stretch", theme=None, key=key or f"chart_{_CURRENT['n']}")
    if table:
        df = fig_to_df(fig)
        if df is not None and len(df):
            with st.expander("Table view", icon=":material/table:"):   # translated by the i18n hook
                shown = df.head(120)
                num = {c: "{:,.2f}" for c in shown.columns if pd.api.types.is_numeric_dtype(shown[c])}
                shown = shown.copy()
                shown.columns = [str(c) for c in shown.columns]
                table_html(shown, num)
                if len(df) > 120:
                    st.caption(tr(f"Showing the first 120 of {len(df):,} rows. Download for the full data."))
                st.download_button("Download CSV", df.to_csv().encode("utf-8"), file_name=f"{_slug(title or 'chart')}.csv",
                                   mime="text/csv", key=f"dl_{_CURRENT['n']}", icon=":material/download:")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-")[:48] or "chart"


# --- page chrome ---------------------------------------------------------------------------------------------------------
def _init_state() -> None:
    """First run of a session: theme and language from the URL (?theme=light&lang=bg), else the browser language."""
    ss = st.session_state
    if "_ui_init" in ss:
        return
    qp = st.query_params
    want_lang = (qp.get("lang") or "").lower()
    if want_lang not in ("en", "bg"):
        try:
            want_lang = "bg" if str(getattr(st.context, "locale", "") or "").lower().startswith("bg") else "en"
        except Exception:
            want_lang = "en"
    want_theme = (qp.get("theme") or "").lower()
    ss["ui_lang"] = "BG" if want_lang == "bg" else "EN"
    ss["ui_theme"] = "Light" if want_theme == "light" else "Dark"
    ss["_ui_init"] = True


def bootstrap() -> None:
    """Install the translation hooks and resolve this session's language and theme (called by the entry script and every page)."""
    from i18n import install_hooks
    install_hooks()
    _init_state()
    ss = st.session_state
    ss["_lang"] = "bg" if ss.get("ui_lang") == "BG" else "en"
    ss["_mode"] = "light" if ss.get("ui_theme") == "Light" else "dark"


def inject_css() -> None:
    css = (ROOT / "app" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}{_mode_css()}</style>", unsafe_allow_html=True)


def _share_url_params() -> None:
    ss = st.session_state
    for k, v in (("lang", ss["_lang"]), ("theme", ss["_mode"])):      # keep the URL shareable
        if st.query_params.get(k) != v:
            st.query_params[k] = v


def chrome() -> None:
    """Brand mark at the left of the header and a Settings popover at the right (theme, language, data status, sign-out)."""
    import auth
    with st.container(key="brandbar"):
        st.markdown("<div class='brand' id='top'><div class='brand-mark' aria-hidden='true'>⚡</div><div><div class='brand-name'>BG Power</div><div class='brand-sub'>SDAC · EET</div></div></div>", unsafe_allow_html=True)
    with st.container(key="settings"):
        with st.popover(tr("Settings"), icon=":material/tune:"):
            st.markdown(f"<div class='pop-label'>{tr('Theme')}</div>", unsafe_allow_html=True)
            st.segmented_control("Theme", ["Dark", "Light"], key="ui_theme", default="Dark", label_visibility="collapsed",
                                 help="Dark or light colours. Charts use a separate, validated palette for each.", width="stretch")
            st.markdown(f"<div class='pop-label'>{tr('Language')}</div>", unsafe_allow_html=True)
            st.segmented_control("Language", ["EN", "BG"], key="ui_lang", default="EN", label_visibility="collapsed",
                                 help="Interface language. Computed numbers and chart data are the same in both.", width="stretch")
            m = load_market()
            through = tr(f"Data through {m.index.max():%d %b %Y}") if m is not None else tr("No data loaded")
            st.markdown(f"<div class='side-meta' role='status'><span class='dot' aria-hidden='true'></span>{through}<br>"
                        f"ENTSO-E · Open-Meteo<br><span class='side-hint'>{tr('Every chart has a <b>Table view</b> and CSV download.')}</span></div>",
                        unsafe_allow_html=True)
            if auth.enabled() and auth.user():
                st.caption(f"{tr('Signed in as')} {auth.user()}")
                if st.button("Sign out", key="signout", icon=":material/logout:", width="stretch"):
                    auth.logout()
                    st.rerun()


def setup(title: str, icon: str = "⚡") -> None:
    bootstrap()
    st.set_page_config(page_title=f"{tr(title)} · BG Power", page_icon=icon, layout="wide")
    inject_css()
    _CURRENT.update(title="", subtitle="", n=0)
    _share_url_params()
    chrome()


def login_page(attempt, multi_user: bool, locked: float) -> None:
    """The sign-in screen. `attempt(user, password)` returns an error message or None when the visitor is in."""
    st.set_page_config(page_title=f"{tr('Sign in')} · BG Power", page_icon="⚡", layout="centered", initial_sidebar_state="collapsed")
    inject_css()
    _share_url_params()
    with st.container(key="login"):
        st.markdown(f"<div class='login-brand'><div class='brand-mark' aria-hidden='true'>⚡</div></div>"
                    f"<div class='login-pill'>ENTSO-E · Open-Meteo · BG</div>"
                    f"<h1 class='login-title'>{tr('Welcome back')}</h1>"
                    f"<p class='login-sub'>{tr('Sign in to open the Bulgarian day-ahead power dashboard.')}</p>", unsafe_allow_html=True)
        with st.form("login_form", border=False):
            user = st.text_input("Username", key="login_user", autocomplete="username") if multi_user else ""
            pw = st.text_input("Password", type="password", key="login_pw", autocomplete="current-password")
            go_ = st.form_submit_button("Sign in", type="primary", width="stretch", disabled=locked > 0)
        if locked > 0:
            st.error(tr(f"Too many attempts. Try again in {int(locked) + 1} s."))
        elif go_:
            err = attempt(user, pw)
            if err is None:
                st.rerun()
            st.error(tr(err))
        st.markdown("<hr class='login-sep'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.segmented_control("Theme", ["Dark", "Light"], key="ui_theme", default="Dark", label_visibility="collapsed", width="stretch")
        c2.segmented_control("Language", ["EN", "BG"], key="ui_lang", default="EN", label_visibility="collapsed", width="stretch")
        st.markdown(f"<p class='login-foot'>{tr('Forecasts and analysis only. This is not trading or investment advice.')}</p>", unsafe_allow_html=True)


def _mode_css() -> str:
    """CSS variables for the current mode, plus the widget overrides the light theme needs (Streamlit itself stays dark)."""
    p = PALETTES[mode()]
    ar = ",".join(str(int(p["ACCENT_UI"].lstrip("#")[i:i + 2], 16)) for i in (0, 2, 4))
    v = dict(bg=p["BG"], surface=p["SURFACE"], surface_2=p["SURFACE_2"], border=p["BORDER"], text=p["TEXT"], text_2=p["TEXT_2"],
             muted=p["MUTED"], accent_ui=p["ACCENT_UI"], accent=p["YELLOW"], up=p["RED"], down=p["BLUE"])
    root = ":root{" + "".join(f"--{k.replace('_', '-')}:{x};" for k, x in v.items()) + f"--accent-rgb:{ar};}}"
    if mode() == "dark":
        return (root + ":root{--border-soft:#1A2230;--sidebar-bg:#090C11;--controls-bg:rgba(18,23,34,.55);--link:#8DB8F2;--shadow:rgba(0,0,0,.18);"
                "--chip-red:#F5A9A9;--chip-blue:#9CC3F5;--chip-green:#6FDB6F;--glow-a:.055;--glow-b:.045;--focus:0 0 0 3px rgba(245,165,36,.55);}")
    return (root + ":root{--border-soft:#E4E0D6;--sidebar-bg:#EAE6DC;--controls-bg:rgba(253,252,250,.7);--link:#1F5FBF;--shadow:rgba(60,50,30,.08);"
            "--chip-red:#A12828;--chip-blue:#1B4F9C;--chip-green:#0B6B0B;--glow-a:.10;--glow-b:.06;--focus:0 0 0 3px rgba(166,74,6,.5);"
            "--good:#0a7a0a;--warn:#8a5a00;--crit:#b02a2a;color-scheme:light;}"
            + (ROOT / "app" / "style_light.css").read_text(encoding="utf-8"))


def header(eyebrow: str, title: str, subtitle: str = "", badges: list[str] | None = None, headline: str = "") -> None:
    """Page header. `headline` is the one-sentence answer for the page (computed from data), shown first."""
    b = "".join(f"<span class='badge'>{html.escape(tr(x))}</span>" for x in badges or [])
    hl = f"<div class='headline' role='note'><span class='hl-tag'>{tr('Headline')}</span><span>{tr(headline)}</span></div>" if headline else ""
    st.markdown(f"<header class='page-head'><div class='eyebrow'>{tr(eyebrow)}</div><h1>{tr(title)}</h1>"
                f"<p class='lede'>{tr(subtitle)}</p>{hl}<div class='badges'>{b}</div></header>", unsafe_allow_html=True)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def section(title: str, subtitle: str = "", anchor: str | None = None) -> None:
    st.markdown(f"<section class='section' id='{anchor or _slug(title)}'><h2>{tr(title)}</h2>" + (f"<p>{tr(subtitle)}</p>" if subtitle else "") + "</section>",
                unsafe_allow_html=True)


def toc(items: list[tuple[str, str]]) -> None:
    """Sticky 'On this page' bar: (anchor, label) pairs matching `section(..., anchor=...)`."""
    links = "".join(f"<a href='#{a}'>{tr(label)}</a>" for a, label in items)
    top_link = f"<a href='#top' class='toc-top' title='{tr('Back to top')}' aria-label='{tr('Back to top')}'>↑</a>"
    with st.container(key="toc"):
        st.markdown(f"<nav class='toc' aria-label='{tr('On this page')}'><span class='toc-label'>{tr('On this page')}</span>{links}{top_link}</nav>", unsafe_allow_html=True)


def link(path: str, label: str, icon: str = ":material/arrow_forward:", **kw) -> None:
    """st.page_link that degrades to nothing when the page is rendered outside the navigation (tests, direct runs)."""
    try:
        st.page_link(path, label=tr(label), icon=icon, width="stretch", **kw)
    except Exception:       # StreamlitPageNotFoundError when there is no st.navigation
        pass


def next_steps(items: list[tuple[str, str, str]], title: str = "Where next") -> None:
    """Related-page links at the foot of a page: (page path relative to the entry script, label, one-line reason)."""
    st.markdown(f"<section class='section'><h2>{tr(title)}</h2></section>", unsafe_allow_html=True)
    with st.container(key="nextsteps"):
        cols = st.columns(len(items))
        for col, (path, label, why) in zip(cols, items):
            with col:
                link(path, tr(label))
                st.caption(why)


_card_n = 0


@contextlib.contextmanager
def card(title: str = "", subtitle: str = "", right: str = "", help: str = ""):
    """Bordered surface. Title should state the finding or the question; subtitle says how to read it. `help` adds an (i) tip."""
    global _card_n
    _card_n += 1
    title, subtitle, right, help = tr(title), tr(subtitle), tr(right), tr(help)
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
        st.markdown(f"<div class='controls-label'>{tr(label)}</div>", unsafe_allow_html=True)
        yield


def sparkline(values, color: str | None = None, w: int = 120, h: int = 34) -> str:
    color = color or P.ACCENT
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
            f"<circle cx='{xs[-1]:.1f}' cy='{ys[-1]:.1f}' r='4' fill='{color}' stroke='{P.SURFACE}' stroke-width='2'/></svg>")


def delta_chip(delta: float | None, unit: str = "%", good_when_up: bool | None = None, digits: int = 1) -> str:
    """Signed change with an arrow AND a word for assistive tech (colour is never the only cue)."""
    if delta is None or not np.isfinite(delta):
        return ""
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "■"
    word = tr("up") if delta > 0 else tr("down") if delta < 0 else tr("flat")
    cls = "flat"
    if good_when_up is not None and delta != 0:
        cls = "pos" if (delta > 0) == good_when_up else "neg"
    elif delta != 0:
        cls = "up" if delta > 0 else "down"
    return f"<span class='chip {cls}'><span aria-hidden='true'>{arrow}</span><span class='sr-only'>{word} </span>{abs(delta):,.{digits}f}{unit}</span>"


def kpi(col, label: str, value: str, sub: str = "", accent: bool = False, spark=None,
        spark_color: str | None = None, delta: str = "", help: str = "") -> None:
    sp = sparkline(spark, spark_color or (P.ACCENT if accent else P.TEXT_2)) if spark is not None else ""
    label, sub, help = tr(label), tr(sub), tr(help)
    tip = f" title='{html.escape(help, quote=True)}'" if help else ""
    col.markdown(f"<div class='kpi{' accent' if accent else ''}'{tip}><div class='label'>{label}</div>"
                 f"<div class='kpi-val'><span class='value'>{value}</span>{delta}</div>"
                 f"<div class='kpi-bottom'><span class='sub'>{sub}</span>{sp}</div></div>", unsafe_allow_html=True)


def takeaway(text: str, kind: str = "insight") -> None:
    icon = {"insight": "◆", "action": "→", "warning": "!"}[kind]
    label = tr({"insight": "Insight", "action": "Action", "warning": "Caution"}[kind])
    text = tr(text)
    st.markdown(f"<div class='takeaway {kind}' role='note'><span class='ti' aria-hidden='true'>{icon}</span>"
                f"<div><span class='sr-only'>{label}: </span>{text}</div></div>", unsafe_allow_html=True)


def legend_html(items: list[tuple[str, str]], kind: str = "line") -> str:
    return "".join(f"<span class='lg'><i class='{kind}' style='background:{c}'></i>{html.escape(tr(n))}</span>" for n, c in items)


def table_html(df: pd.DataFrame, fmt: dict | None = None, bars: dict | None = None, highlight: str | None = None,
               badges: dict | None = None) -> None:
    fmt, bars, badges = fmt or {}, bars or {}, badges or {}
    cols = list(df.columns)
    head = f"<th scope='col'>{html.escape(str(df.index.name or ''))}</th>" + "".join(f"<th scope='col'>{html.escape(tr(str(c)))}</th>" for c in cols)
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
    st.markdown(f"<div class='tbl-wrap{' tall' if len(rows) > 12 else ''}' role='region' tabindex='0' aria-label='{tr("data table")}'><table class='tbl'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


table = table_html


def missing(name: str) -> None:
    st.markdown(f"<div class='empty' role='status'><div class='empty-icon' aria-hidden='true'>◌</div><div><b>{tr(name)}</b> {tr('is not available yet.')}<br>"
                f"<span>{tr('Run <code>python -m src.run_all --plus</code> to generate the reports.')}</span></div></div>", unsafe_allow_html=True)


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
    return P.GEN_GROUP_COLOR.get(gen_group(name), P.NEUTRAL)


def gen_label(name: str) -> str:
    return GEN_LABEL.get(name, name.replace("_", " ").capitalize())


def fmt_eur(x: float, d: int = 1) -> str:
    """Euro amount with a proper minus sign in front of the currency symbol (−€79, not €-79)."""
    return f"{'−' if x < 0 and round(abs(x), d) > 0 else ''}€{abs(x):,.{d}f}"
