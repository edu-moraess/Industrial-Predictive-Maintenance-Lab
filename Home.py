"""
Industrial Predictive Maintenance Lab — entry point.

Run from repository root:

    streamlit run Home.py

Pages appear automatically in the Streamlit sidebar:
  - Operations (sensor ML lab)
  - Computer Vision (visual inspection lab)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
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
    <div style="border-bottom:1px solid #2A2F38;padding-bottom:12px;margin-bottom:18px;">
        <h1 style="font-size:1.35rem;font-weight:600;margin:0;letter-spacing:0.02em;">
            INDUSTRIAL PREDICTIVE MAINTENANCE LAB
        </h1>
        <p style="color:#9A9FA8;font-size:0.85rem;margin:6px 0 0 0;">
            Synthetic sensor intelligence + experimental computer vision inspection
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="ind-card">
            <div class="ind-card-header">OPERATIONS</div>
            <p style="color:#F2F2F2;font-size:0.9rem;">
                Virtual machine telemetry, Isolation Forest, Random Forest,
                health score and experimental RUL.
            </p>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:0;">
                Open <strong>Operations</strong> in the left sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="ind-card">
            <div class="ind-card-header">COMPUTER VISION</div>
            <p style="color:#F2F2F2;font-size:0.9rem;">
                Image / video upload, object detection, tracking,
                baseline visual anomaly (heuristic).
            </p>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:0;">
                Open <strong>Computer Vision</strong> in the left sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Sensor pipeline and vision pipeline are independent. "
    "Synthetic data \u00b7 experimental models \u00b7 not industrial certification."
)
