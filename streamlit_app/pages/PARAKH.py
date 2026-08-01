import os
import sys

import streamlit as st

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

st.set_page_config(page_title="PARAKH — DataTalk", layout="wide",
                   page_icon=os.path.join(_APP, "images", "logo.png")
                   if os.path.exists(os.path.join(_APP, "images", "logo.png"))
                   else "🧭")
try:
    st.logo(os.path.join(_APP, "images", "logo.png"), size="large")
except Exception:
    pass

import mission_intel

mission_intel.render_parakh()
