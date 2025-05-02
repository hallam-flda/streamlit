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

st.header("The Maths Behind the Angle", divider=True)


st.subheader("Expected Goals")
st.write(
"""
In order to calculate the probability of a Centre Back being assisted by a Set Piece Taker we need to compute
a few probabilities. The first and most obvious is the probability of the Centre Back in question scoring at all.

In general, goalscorer prices are generated as a function of expected number of goals their team is expected to score
and then the proportion of these goals that we expect to be scored by the player.

First we need to get the number of goals we expect there to be in a game. There are a few ways to do this, one of which is to create
a power-rating based on the relative home and away performance of the two teams in question.

Since I have scraped the league table from FBref, split by home and away performance, we can take each team's respective average xG for and xG against
and combine them to calculate an expected scoreline.
"""
)

with st.echo():
    
    import pandas as pd
    
    prem_table_ha = pd.read_csv("data/data/fbref_dashboard/prem_table_ha.csv")

st.dataframe(prem_table_ha.head())

st.write(
""" 
This table is an exact replica of the league table format on FBref, now we need to do some pre-processing to get the inputs to our Poisson model
"""
)

with st.echo():
    
    def team_rating_cols(df):
        df['Home_xGp90'] = df['Home_xG']/df["Home_MP"]
        df['Home_xGAp90'] = df['Home_xGA']/df["Home_MP"]
        df['Away_xGp90'] = df['Away_xG']/df["Away_MP"]
        df['Away_xGAp90'] = df['Away_xGA']/df["Away_MP"]
        df['league_home_xGp90'] = df['Home_xGp90'].mean()
        df['league_home_xGAp90'] = df['Home_xGAp90'].mean()
        df['league_away_xGp90'] = df['Away_xGp90'].mean()
        df['league_away_xGAp90'] = df['Away_xGAp90'].mean()
        df['home_att_rating'] = df['Home_xGp90']/df['league_home_xGp90']
        df['home_def_rating'] = df['Home_xGAp90']/df['league_home_xGAp90']
        df['away_att_rating'] = df['Away_xGp90']/df['league_away_xGp90']
        df['away_def_rating'] = df['Away_xGAp90']/df['league_away_xGAp90']
        return df

    rated_team_table = team_rating_cols(prem_table_ha)
    
st.dataframe(rated_team_table)

st.write(
"""
Now we have a league standardised rating for each team's performance at home and away, we can use these to model
the number of goals we expected a fixture to have.

_Note, this only works after a sufficient number of games have been played in order for
the average performance to make sense. If we were to use this system after the first 3 games of the season, there would be too high of a bias
placed on the quality of teams faced._
"""
)


st.subheader("Work Log", divider = True)
st.markdown(
"""
- Only render probability after all previous elements showing ✅ 
- Reverse order of butterfly chart for left hand side ✅ 
- Change Radar chart to only compare against each other not league average. ✅ 
    - Maybe change to use radar chart from mplsoccer package ❌ - didn't look good
    - Change metrics? want user to know likelihood of taking set pieces
- Filter out non-CBs from defender list
- Formalise function for plotting team lineups 
    - Test with prem data for previous seasons
- Split tabs into analysing CBs/SPTs/Theory Write-up
"""
)
