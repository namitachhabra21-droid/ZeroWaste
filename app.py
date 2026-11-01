"""ZeroWaste: next-day food waste forecasting for real kitchens (URBAN-WASTE records)."""

import streamlit as st

from real_app import render_real_app

st.set_page_config(page_title="ZeroWaste Forecast", page_icon="🍲", layout="wide", initial_sidebar_state="expanded")
st.title("🍲 ZeroWaste · Food Waste Forecasting")
render_real_app()
