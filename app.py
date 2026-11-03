"""ZeroWaste: next-day food waste forecasting for real kitchens (URBAN-WASTE records)."""

import streamlit as st

from real_app import render_real_app

# Streamlit entry point: configure the page, then delegate UI rendering.
# Training is performed separately; the dashboard loads a saved forecasting artifact.
st.set_page_config(page_title="ZeroWaste Forecast", page_icon="🍲", layout="wide", initial_sidebar_state="expanded")
st.title("🍲 ZeroWaste · Food Waste Forecasting")
render_real_app()
