"""
Legacy entry: streamlit run app/dashboard.py

Preferred entry (multipage with Computer Vision page):

    streamlit run Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.styles import INDUSTRIAL_THEME_CSS
from app.operations_page import render_operations

st.set_page_config(
    page_title="Industrial Operations Center",
    page_icon="\u2699\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(INDUSTRIAL_THEME_CSS, unsafe_allow_html=True)

st.info(
    "For **Computer Vision** page navigation, run from repo root: `streamlit run Home.py`"
)
render_operations()
