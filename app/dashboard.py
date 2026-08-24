"""
Industrial Predictive Maintenance Lab — Streamlit entry.

    streamlit run app/dashboard.py

Top selector switches between Operations and Computer Vision.
Only the selected module runs each cycle (avoids dual sidebar / rerun conflicts).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.styles import INDUSTRIAL_THEME_CSS

st.set_page_config(
    page_title="Industrial Predictive Maintenance Lab",
    page_icon="\u2699\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(INDUSTRIAL_THEME_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div style="border-bottom:1px solid #2A2F38;padding-bottom:8px;margin-bottom:12px;">
        <h1 style="font-size:1.05rem;font-weight:600;margin:0;">INDUSTRIAL PREDICTIVE MAINTENANCE LAB</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Module",
    ["Operations", "Computer Vision"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_module_selector",
)

st.markdown("---")

if mode == "Computer Vision":
    try:
        from app.vision_page import render_computer_vision

        render_computer_vision()
    except Exception as exc:  # noqa: BLE001 — show error instead of blank page
        st.error("Computer Vision failed to load.")
        st.exception(exc)
        st.info(
            "If packages are missing, run: `pip install opencv-python-headless ultralytics`"
        )
else:
    from app.operations_page import render_operations

    render_operations()
