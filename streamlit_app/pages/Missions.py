"""🎯 National Missions — separate page (light design preserved).

Appears as a nav button in the sidebar automatically (Streamlit pages/
mechanism). Reuses missions_page.py unchanged. Reads the primary dataset
the user uploaded on the main dashboard via session_state; if the user
lands here first, Sections 2-3 explain what to upload on the main page.
"""
import pathlib
import sys

import streamlit as st

st.set_page_config(page_title="National Missions", page_icon="🎯",
                   layout="wide")

_APP_DIR = pathlib.Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import missions_page as _mp                                    # noqa: E402

_mp._STANDALONE = True          # our full light design: hero, cards, CSS

_df = st.session_state.get("_mx_primary_df")
_score = st.session_state.get("_mx_score_col")
_qmap = st.session_state.get("_mx_qmap")

if _df is None:
    st.sidebar.info("No primary dataset in this session yet — upload it on "
                    "the main dashboard page, then return here. Section 1 "
                    "works without it.")

_mp.render(df=_df, score_col=_score, qmap=_qmap)