"""Shared visual language for the app: a validated dark palette (categorical,
sequential, diverging, status), a matching Plotly template, and the CSS that
gives the dashboard its look. One source of truth so every page reads as the
same system."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Categorical (fixed order — never cycle/reassign)
SERIES = {
    "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500",
    "magenta": "#d55181", "green": "#008300", "violet": "#9085e9", "red": "#e66767",
}
CATEGORICAL = [SERIES["blue"], SERIES["orange"], SERIES["aqua"], SERIES["yellow"],
               SERIES["magenta"], SERIES["green"], SERIES["violet"], SERIES["red"]]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"cold": "#3987e5", "mid": "#383835", "warm": "#e66767"}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
BORDER = "rgba(255,255,255,0.10)"


def register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_SECONDARY, size=13),
        title=dict(font=dict(color=INK_PRIMARY, size=15)),
        colorway=CATEGORICAL,
        xaxis=dict(gridcolor=GRIDLINE, linecolor=BASELINE, zerolinecolor=BASELINE,
                    tickfont=dict(color=INK_MUTED), title_font=dict(color=INK_SECONDARY)),
        yaxis=dict(gridcolor=GRIDLINE, linecolor=BASELINE, zerolinecolor=BASELINE,
                    tickfont=dict(color=INK_MUTED), title_font=dict(color=INK_SECONDARY)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SECONDARY)),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor="#242422", font=dict(color=INK_PRIMARY), bordercolor=BORDER),
    )
    pio.templates["trading_dark"] = template
    pio.templates.default = "trading_dark"


def inject_css() -> None:
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {PAGE}; }}
        section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
        [data-testid="stMetric"] {{
            background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricLabel"] {{ color: {INK_MUTED}; }}
        div.block-container {{ padding-top: 2rem; max-width: 1400px; }}
        .app-header {{
            display: flex; align-items: center; justify-content: space-between;
            padding-bottom: 6px; margin-bottom: 18px; border-bottom: 1px solid {BORDER};
        }}
        .app-title {{ font-size: 1.5rem; font-weight: 700; color: {INK_PRIMARY}; }}
        .app-subtitle {{ color: {INK_MUTED}; font-size: 0.85rem; }}
        .stat-card {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: 16px 18px; height: 100%;
        }}
        .stat-card .label {{ color: {INK_MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
        .stat-card .value {{ color: {INK_PRIMARY}; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
        .stat-card .sub {{ color: {INK_SECONDARY}; font-size: 0.8rem; margin-top: 2px; }}
        .badge {{
            display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem;
            font-weight: 600; letter-spacing: 0.02em;
        }}
        .badge-good {{ background: rgba(12,163,12,0.18); color: #4ade4a; }}
        .badge-warning {{ background: rgba(250,178,25,0.18); color: #fab219; }}
        .badge-serious {{ background: rgba(236,131,90,0.18); color: #ec835a; }}
        .badge-critical {{ background: rgba(208,59,59,0.18); color: #ff6b6b; }}
        .badge-neutral {{ background: rgba(255,255,255,0.08); color: {INK_SECONDARY}; }}
        .section-title {{ color: {INK_PRIMARY}; font-size: 1.05rem; font-weight: 600; margin: 8px 0 10px 0; }}
        hr {{ border-color: {BORDER}; }}
        thead tr th {{ color: {INK_MUTED} !important; }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div class="app-header">
        <div>
            <div class="app-title">{title}</div>
            <div class="app-subtitle">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def apply_theme(title: str, subtitle: str = "") -> None:
    inject_css()
    register_plotly_template()
    page_header(title, subtitle)
