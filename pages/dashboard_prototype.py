import streamlit as st
import pandas as pd
from utils import fbref

st.title("Centre Back to be Assisted by Set Piece Taker")
st.write("")

st.write(
"""
Following a conversation with some former colleagues, I wanted to investigate if there is a pricing inefficiency
with combining Centre-backs to score assisted by a set piece taker. Initial analysis has shown that on average, if a centre back scores,
there is 47% chance it was assisted by one of the club's set-piece takers in that season.

I took this as an opportunity to practise dashboarding using streamlit, this is still a work in progress but can be viewed
[here](https://hallam-flda-cb-angle.streamlit.app/)
"""
)

st.caption("Dashboard Preview")
st.image("media/fb_dashboard/fb_dashboard.png")

st.subheader("Work Log", divider = True)
st.markdown(
"""
- Only render probability after all previous elements showing
- Reverse order of butterfly chart for left hand side
- Change Radar chart to only compare against each other not league average.
    - Maybe change to use radar chart from mplsoccer package
    - Change metrics? want user to know likelihood of taking set pieces
- Filter out non-CBs from defender list
- Formalise function for plotting team lineups
    - Test with prem data for previous seasons
- Split tabs into analysing CBs/SPTs/Theory Write-up
"""
)
