"""Computer Vision Inspection — independent visual lab."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.styles import INDUSTRIAL_THEME_CSS
from app.vision_page import render_computer_vision

st.markdown(INDUSTRIAL_THEME_CSS, unsafe_allow_html=True)
render_computer_vision()
