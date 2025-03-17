import streamlit as st
import streamlit.components.v1 as components

st.title("Garmin Activity Map")
st.caption("An interactive plot of all my recorded Garmin activities from 01/01/2023 - 15/02/2025")

kepler_map_path = "static/activity_kepler.html"

with open(kepler_map_path, "r", encoding="utf-8") as f:
    html_string = f.read()

# Embed in Streamlit
components.html(html_string, height=600, width=1000, scrolling=True)