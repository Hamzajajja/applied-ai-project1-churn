import plotly.graph_objects as go
import streamlit as st

from config import COLORS

_cfg = {"cur": "PKR", "fx": 280.0, "theme": "dark"}
_counter = {"n": 0}


def configure(cur: str, fx: float, theme: str) -> None:
    _cfg.update(cur=cur, fx=fx, theme=theme)


def scale() -> float:
    return _cfg["fx"] if _cfg["cur"] == "PKR" else 1.0


def prefix() -> str:
    return "Rs " if _cfg["cur"] == "PKR" else "$"


def money(usd: float, compact: bool = False) -> str:
    """Formats a USD amount in the selected currency. PKR uses Lac / Cr like people actually say it."""
    value = usd * scale()
    sign = "-" if value < 0 else ""
    v = abs(value)
    if compact:
        if _cfg["cur"] == "PKR":
            if v >= 1e7:
                return f"{sign}Rs {v / 1e7:,.2f} Cr"
            if v >= 1e5:
                return f"{sign}Rs {v / 1e5:,.2f} Lac"
        else:
            if v >= 1e6:
                return f"{sign}${v / 1e6:,.2f}M"
            if v >= 1e3:
                return f"{sign}${v / 1e3:,.1f}K"
    return f"{sign}{prefix()}{v:,.0f}"


def reset_keys() -> None:
    _counter["n"] = 0


def style(fig: go.Figure, height: int = 380, title: str = None) -> go.Figure:
    template = "plotly_dark" if _cfg["theme"] == "dark" else "plotly_white"
    fig.update_layout(
        template=template, height=height, title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def show(fig: go.Figure, height: int = 380, title: str = None) -> None:
    _counter["n"] += 1
    st.plotly_chart(style(fig, height, title), width="stretch",
                    config={"displaylogo": False}, key=f"plot_{_counter['n']}")


def table(df, **kwargs) -> None:
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)


def kpi(col, label: str, value: str, sub: str = "", color: str = None) -> None:
    color = color or COLORS["accent"]
    col.markdown(
        f"<div class='kpi' style='border-top:3px solid {color};'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value' style='color:{color};'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def insight(col, icon: str, text: str) -> None:
    col.markdown(f"<div class='insight'><span class='ico'>{icon}</span><span>{text}</span></div>",
                 unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list) -> None:
    chip_html = "".join(f"<span class='chip'>{c}</span>" for c in chips)
    st.markdown(
        f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p><div>{chip_html}</div></div>",
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    st.markdown(f"### {title}")
    if note:
        st.markdown(f"<p class='note'>{note}</p>", unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
            .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }

            .hero {
                background: linear-gradient(120deg, #0f766e 0%, #1d4ed8 55%, #7c3aed 100%);
                border-radius: 18px; padding: 26px 32px; margin-bottom: 18px; color: #fff;
                box-shadow: 0 10px 30px rgba(29, 78, 216, .25);
            }
            .hero h1 { margin: 0 0 4px 0; font-size: 1.85rem; font-weight: 800; color: #fff; }
            .hero p  { margin: 0 0 14px 0; opacity: .88; font-size: .95rem; color: #fff; }
            .chip {
                display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
                padding: 3px 12px; border-radius: 999px; font-size: .76rem; margin: 0 8px 4px 0; color: #fff;
            }

            .kpi {
                background: linear-gradient(145deg, rgba(128,128,128,.12), rgba(128,128,128,.04));
                border: 1px solid rgba(128,128,128,.22); border-radius: 14px;
                padding: 16px 18px; height: 100%;
                transition: transform .18s ease, box-shadow .18s ease;
            }
            .kpi:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,.18); }
            .kpi .label { font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; opacity: .65; }
            .kpi .value { font-size: 1.65rem; font-weight: 800; line-height: 1.3; margin-top: 2px; }
            .kpi .sub   { font-size: .76rem; opacity: .6; }

            .insight {
                display: flex; gap: 10px; align-items: flex-start; font-size: .84rem; line-height: 1.45;
                background: rgba(56,189,248,.08); border: 1px solid rgba(56,189,248,.25);
                border-radius: 12px; padding: 12px 14px; height: 100%;
            }
            .insight .ico { font-size: 1.15rem; }

            .note { font-size: .82rem; opacity: .7; margin-top: -6px; }
            .action-card {
                border-radius: 14px; padding: 16px 20px; margin-top: 6px;
                background: linear-gradient(120deg, rgba(34,197,94,.14), rgba(56,189,248,.10));
                border: 1px solid rgba(34,197,94,.35);
            }
            .action-card b { font-size: 1.02rem; }

            button[data-baseweb="tab"] { font-weight: 600; font-size: .92rem; }
            section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
