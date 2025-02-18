import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

sheet_id = '1q1xDAhCBlIs9VBPu698HSCysR72GWLJRu_0EBogYTgk'
csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'

df = pd.read_csv(csv_url)

# Lag the values from the previous entry in database
df['spanish_lagged'] = df.spanish.shift(1)
df['portuguese_lagged'] = df.portuguese.shift(1)

# Calculate difference with previous values
df['spanish_xp_change'] = df.spanish - df.spanish_lagged
df['portuguese_xp_change'] = df.portuguese - df.portuguese_lagged
df['date'] = pd.to_datetime(df['date']).dt.date


fig = px.bar(df, x="date", y=["spanish_xp_change","portuguese_xp_change"], title="Daily XP Change by Language")

st.title("Hello This is a new page")
st.header("Just checking this isn't the reason a new window opens every time")
st.plotly_chart(fig)