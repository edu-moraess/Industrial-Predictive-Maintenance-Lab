"""
Industrial Predictive Maintenance Lab — Streamlit entry.

Works with either:

    streamlit run app/dashboard.py
    streamlit run Home.py

Computer Vision is always reachable from this file via top-level tabs.
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

# Top-level module switch — visible without multipage / Home.py
tab_ops, tab_cv = st.tabs(["Operations", "Computer Vision"])

with tab_ops:
    from app.operations_page import render_operations

    render_operations()

with tab_cv:
    from app.vision_page import render_computer_vision

    render_computer_vision()
