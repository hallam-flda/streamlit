
import streamlit as st
import pandas as pd
from utils import fbref

home_squad_df = pd.read_csv("data/data/fbref_dashboard/palace_squad_24_25.csv")
home_squad_df['team'] = 'Crystal Palace'
 
# prem_table = pd.read_csv("data/data/fbref_dashboard/prem_table.csv")
# st.dataframe(prem_table.iloc[:,1:-4])

prem_table_ha = pd.read_csv("data/data/fbref_dashboard/prem_table_ha.csv")
team_list = set(prem_table_ha.Squad)

home_team = st.selectbox(
    "Home Team",
    team_list,
    index=None,
    placeholder="Select Home Team...",
)

if home_team:
    home_stats = prem_table_ha[prem_table_ha["Squad"] == home_team]
    st.dataframe(home_stats.iloc[:,1:])

away_team = st.selectbox(
    "Away Team",
    team_list,
    index=None,
    placeholder="Select Away Team...",
)

if away_team:
    away_stats = prem_table_ha[prem_table_ha["Squad"] == away_team]
    st.dataframe(away_stats.iloc[:,1:])

home_col, away_col, output_probs = st.columns([1,1,2])

with home_col:
    st.dataframe(home_squad_df)