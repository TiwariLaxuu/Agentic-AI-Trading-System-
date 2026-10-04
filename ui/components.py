"""Small reusable HTML components (stat tiles, badges) built on the theme
tokens, plus a couple of formatting helpers shared across pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

STATUS_CLASS = {
    "good": "badge-good", "warning": "badge-warning",
    "serious": "badge-serious", "critical": "badge-critical", "neutral": "badge-neutral",
}


def badge_html(text: str, status: str = "neutral") -> str:
    cls = STATUS_CLASS.get(status, "badge-neutral")
    return f'<span class="badge {cls}">{text}</span>'

def badge(text: str, status: str = "neutral") -> None:
    st.markdown(badge_html(text, status), unsafe_allow_html=True)


def stat_card(label: str, value: str, sub: str = "", status: str | None = None) -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="stat-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """


def stat_row(items: list[dict]) -> None:
    """items: [{label, value, sub, status}]"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(stat_card(item.get("label", ""), item.get("value", ""), item.get("sub", "")),
                        unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def fmt_pct(x, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return "—"
    except Exception:
        pass
    return f"{x:+.{digits}f}%"


def fmt_num(x, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return "—"
    except Exception:
        pass
    return f"{x:,.{digits}f}"


def format_df(df: pd.DataFrame, fmts: dict) -> pd.DataFrame:
    """Pre-format numeric columns to display strings without pandas Styler
    (which pulls in matplotlib — avoided here since it isn't a reliable
    dependency across environments). Pass the result straight to
    st.dataframe instead of chaining `.style.format(...)`."""
    out = df.copy()
    for col, fmt in fmts.items():
        if col not in out.columns:
            continue
        out[col] = out[col].map(lambda v: (fmt.format(v) if pd.notna(v) else "—") if not isinstance(v, str) else v)
    return out


def regime_status(vol_regime: str) -> str:
    return {"Low": "good", "Normal": "neutral", "High": "warning", "Extreme": "critical"}.get(vol_regime, "neutral")
