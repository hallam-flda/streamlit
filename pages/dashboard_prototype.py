
import streamlit as st
import pandas as pd
from utils import fbref

squad_df = pd.read_csv("data/data/fbref_dashboard/palace_squad_24_25.csv")
squad_df['team'] = 'Crystal Palace'
 
team_list = set(squad_df.team)

home_col, away_col, output_probs = st.columns([1,1,2])


with home_col:
    
    home_team = st.selectbox(
        "Home Team",
        team_list,
        index=None,
        placeholder="Select Home Team...",
    )

    home_formation = st.selectbox(
        "Home Formation",
        ("3-4-3"),
        index = None,
        placeholder="Select Home Formation..."
        )


test_func = fbref.most_common_team(squad_df)
st.write(test_func)