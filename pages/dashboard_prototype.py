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

st.subheader("Introduction")
st.write(
"""
The probability we are attempting to estimate here is the probability of a given Centre-Back scoring a goal but a given Set-Piece Taker. I have done this using
a poisson model taking inputs for:
1. Expected Goals
2. Centre Back share of xG
3. Likelihood assist is from Set-Piece Taker (47%)
4. and share of set-pieces taken by set-piece taker.

There are numerous improvements that can be made to this model - however, the purpose of this exercise was to practice my dashboarding in Streamlit. Further down the line I would
like to refine the model by improving the Centre-Back's share of goals and the type of set-pieces taken.

More formally the model can be expressed as such:
"""
)

st.markdown(r"""
1. $\lambda_{h}, \lambda_{a}$ for expected goals home and away
2. $\lambda_{CB_h} = \frac{xGp90_{CB}}{xGp90_{Team}} * \lambda_{h}$     and      $\lambda_{CB_a} = \frac{xGp90_{CB}}{xGp90_{Team}} * \lambda_{a}$
3. $P_{assist} = 0.47 * $ proportion of set pieces taken by set piece taker (defined by a slider in the dashboard)
            
and so the final probability that a named CB will score assisted by a set-piece taker is 
            
$$
P = 1-exp(-P_{assist}*\lambda_{CB})       
$$     


I will work through an example with all associated code below using the fixture of Crystal Palace at home (my team) vs Brighton (our rivals).
""", unsafe_allow_html=True)

st.subheader("Expected Goals")
# In order to calculate the probability of a Centre Back being assisted by a Set Piece Taker we need to compute
# a few probabilities. The first and most obvious is the probability of the Centre Back in question scoring at all.

# In general, goalscorer prices are generated as a function of expected number of goals their team is expected to score
# and then the proportion of these goals that we expect to be scored by the player.
st.write(
"""
First we need to get the number of goals we expect there to be in a game. There are a few ways to do this, one of which is to create
a power-rating based on the relative home and away performance of the two teams in question.

Since I have scraped the league table from FBref, split by home and away performance, we can take each team's respective average xG for and xG against
and combine them to calculate an expected scoreline.
"""
)

with st.echo():
    
    import pandas as pd
    
    prem_table_ha = pd.read_csv("data/data/fbref_dashboard/prem_table_ha.csv")

with st.expander("See scraping function"):
    st.code("""
    def get_league_data(url):
    # home and away form
    league_table_ha_df = pd.read_html(url, header = [0,1])[1]
    league_table_ha_df.columns = [
        f"{parent.strip()}_{child.strip()}" if "Unnamed" not in str(parent) else child.strip()
        for parent, child in league_table_ha_df.columns
    ]

    return league_table_ha_df        
    
    def league_table_update():
    time.sleep(6) # more than 10 requests in a minute will get you blocked from fbref for a day!
    league_table_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    league_table_df = get_league_data(league_table_url)
    league_table_df.to_csv(f"{OUTPUT_PATH}prem_table_ha.csv")
    print('league table data saved')
    """)
    
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

st.subheader("Example")

st.write(
"""
Taking this table I will now define a function for calculating the expected number of goals and plug in our example case of Crystal Palace vs Brighton

"""
)

with st.echo():
    
    def poisson_rating(df, home_team, away_team):
        home_team_df = df[df['Squad'] == home_team]
        away_team_df = df[df['Squad'] == away_team]
        home_team_df = home_team_df.iloc[0]
        away_team_df = away_team_df.iloc[0]

        mu_home = home_team_df['league_home_xGp90']
        att_home = home_team_df['home_att_rating']
        def_away = away_team_df['away_def_rating']

        lambda_home = mu_home * att_home * def_away

        mu_away = away_team_df['league_away_xGp90']
        att_away = away_team_df['away_att_rating']
        def_home = home_team_df['home_def_rating']

        lambda_away = mu_away * att_away * def_home

        return lambda_home, lambda_away

    lambda_palace, lambda_brighton = poisson_rating(rated_team_table, 'Crystal Palace', 'Brighton')
    
st.markdown(f"""
            We now have our expected goals for Palace $\lambda_h$ = {round(lambda_palace,2)} and Brighton $\lambda_a$ = {round(lambda_brighton,2)}

            This is a credible outcome, the [real fixture](https://fbref.com/en/partidos/e9cb51b4/Crystal-Palace-Brighton-and-Hove-Albion-Abril-5-2025-Premier-League) resulted in a 2-1 win for Crystal Palace (and 3 red cards!)
            """)

st.subheader("Spreadex Override")

st.write(
"""
While it's nice to be able to derive this myself, I know that the poisson calculation I have used for this is simplistic and there are better sources of information available.

One of these is [Spreadex](https://www.spreadex.com/sports/en-GB/spread-betting) where we can take the expected number of goals and the home supremacy as the midpoint between
the two figures. In the example below Spreadex anticipate there being 3.75 goals in this weekend's fixture between Southampton and Man City with City scoring 2.15 more goals than Southampton
"""    
)

st.image("media/fb_dashboard/spreadex_city_soton.png")

st.write(
"""
We can then calculate the expected number of goals using these figures
"""    
)

with st.echo():
    home_suprem = -2.15
    total_goals = 3.75
    home_goals = home_suprem/2 + total_goals/2
    away_goals = total_goals - home_goals
    st.write(f"Home Goals: {home_goals:.2f}")  
    st.write(f"Away Goals: {away_goals:.2f}")

st.write(
"""
Compare this to the function I used earlier
"""    
)

with st.echo():
    lambda_soton, lambda_city = poisson_rating(rated_team_table, 'Southampton', 'Manchester City')
    st.write(f"Home Goals: {lambda_soton:.2f}")  
    st.write(f"Away Goals: {lambda_city:.2f}")
    
st.write(
"""
While not a huge difference, this can increase the accuracy of our final output. As such, I have included a manual override for the user of the dashboard to input spreadex values.
"""    
)

st.subheader("Centre Back Goal Proportion")

st.write(
"""
There is nothing fancy going on here, I have simply taken the xGp90 of the Centre Back selected over the whole team xGp90. Again, this could do with a great deal of improvement as it
assumes the player is playing in fixtures representative of how the team always plays.

For a CB that starts every fixture, this should work, however, for one who has only played in 
fixtures where the team has generated lots of shooting opportunities, we will undoubtably be inflating the likelihood of him scoring.
"""
)

with st.echo():
    player_stats = pd.read_csv("data/data/fbref_dashboard/all_prem_squads.csv")


with st.expander("See scraping function"):
    st.code("""
    def get_team_data(url):
        time.sleep(10)
        print(f"trying URL {url}")
        squad_tables = pd.read_html(url)
        league_squad_df = squad_tables[0]

        prefix_map = {
            "Playing Time": "ptime_",
            "Performance": "perf_",
            "Expected": "exp_",
            "Progression": "prog_",
            "Per 90 Minutes": "p90_"
        }


        league_squad_df.columns = [
            f"{prefix_map.get(parent.strip(), '')}{child.strip()}" if "Unnamed" not in str(parent) else child.strip()
            for parent, child in league_squad_df.columns
        ]

        # extracting team name from url
        match = re.search(r'/([^/]+)-Stats$', url)
        team_name = match.group(1).replace('-', ' ') if match else None
        season = squad_tables[1]["Date"][0][:4]

        # for joining later
        league_squad_df['Team'] = team_name
        league_squad_df['Season'] = season

        # last two rows are totals
        league_squad_df = league_squad_df[:-2]

        print(f'{team_name} gathered')
        return league_squad_df


    def squad_data_update():
        squad_urls = [
            "https://fbref.com/en/squads/18bb7c10/Arsenal-Stats",
            "https://fbref.com/en/squads/8602292d/Aston-Villa-Stats",
            "https://fbref.com/en/squads/4ba7cbea/Bournemouth-Stats",
            "https://fbref.com/en/squads/cd051869/Brentford-Stats",
            "https://fbref.com/en/squads/d07537b9/Brighton-and-Hove-Albion-Stats",
            "https://fbref.com/en/squads/cff3d9bb/Chelsea-Stats",
            "https://fbref.com/en/squads/47c64c55/Crystal-Palace-Stats",
            "https://fbref.com/en/squads/d3fd31cc/Everton-Stats",
            "https://fbref.com/en/squads/fd962109/Fulham-Stats",
            "https://fbref.com/en/squads/b74092de/Ipswich-Town-Stats",
            "https://fbref.com/en/squads/a2d435b3/Leicester-City-Stats",
            "https://fbref.com/en/squads/822bd0ba/Liverpool-Stats",
            "https://fbref.com/en/squads/b8fd03ef/Manchester-City-Stats",
            "https://fbref.com/en/squads/19538871/Manchester-United-Stats",
            "https://fbref.com/en/squads/b2b47a98/Newcastle-United-Stats",
            "https://fbref.com/en/squads/a2d435b3/Nottingham-Forest-Stats",
            "https://fbref.com/en/squads/33c895d4/Southampton-Stats",
            "https://fbref.com/en/squads/361ca564/Tottenham-Hotspur-Stats",
            "https://fbref.com/en/squads/7c21e445/West-Ham-United-Stats",
            "https://fbref.com/en/squads/8cec06e1/Wolverhampton-Wanderers-Stats"
        ]

        all_prem_squads = []

        for url in squad_urls:
            print(f'starting {url}')
            df = get_team_data(url)
            all_prem_squads.append(df)
            print('data appended')

        all_prem_squads_df = pd.concat(all_prem_squads)
        all_prem_squads_df.to_csv(f"{OUTPUT_PATH}all_prem_squads.csv")
        print('squad data saved')
    """)

with st.echo():
    st.dataframe(player_stats[(player_stats["Team"] == 'Brighton and Hove Albion') & (player_stats["Pos"] == 'DF')].head())

st.write(
"""
For this example I will use Marc Guehi and Lewis Dunk as both are regular starters for their teams
"""
)

with st.echo():
    
    home_team = 'Crystal Palace'
    away_team = 'Brighton' 
    home_defender = 'Marc Guéhi'
    away_defender = 'Lewis Dunk'
    
    home_team_df = rated_team_table[rated_team_table['Squad'] == home_team].iloc[0]
    home_player_df = player_stats[(player_stats['Player'] == home_defender) & (player_stats['Team'] == home_team)].copy()
    home_player_df['player_xG_contr'] = round(home_player_df['p90_xG'] / home_team_df['Home_xGp90'],3)
    home_player_df = home_player_df.iloc[0]
    home_defender_cont = home_player_df['player_xG_contr']
    

    away_team_df = rated_team_table[rated_team_table['Squad'] == away_team].iloc[0]
    # different name for brighton - dashboard version has function to account for this
    away_player_df = player_stats[(player_stats['Player'] == away_defender) & (player_stats['Team'] == 'Brighton and Hove Albion')].copy()
    away_player_df['player_xG_contr'] = round(away_player_df['p90_xG'] / away_team_df['Away_xGp90'],3)
    away_player_df = away_player_df.iloc[0]
    away_defender_cont = away_player_df['player_xG_contr']
    
    st.write(f"{home_defender}'s average xG goal contribution is {round(home_defender_cont*100,3)}% ")
    st.write(f"{away_defender}'s average xG goal contribution is {round(away_defender_cont*100,3)}% ")
    
st.subheader("Set Piece Takers")

st.write(
"""
There is no mathematical consideration for the model here, simply a list of set-piece takers as determined using a criteria based on the proportion of dead-ball passes they take.
This dataset was taken from FBref and joined onto Transfermarkt data that lists who assisted each goal. Therefore this criteria was written in my GCP BigQuery instance.
"""
)

with st.expander("See SQL Code"):
    st.code(
        """
        create table transfermarkt.fbref_dashboard_spts as 

        WITH set_piece_takers as 
        (
        SELECT
        season,
        player_club,
        player_name,
        position,
        games_played,
        games_played*90 as minutes_played,
        passes_attempted,
        pass_types_dead,
        pass_types_fk,
        pass_types_ck,
        corner_kicks_in,
        corner_kicks_out,
        outcomes_blocks

        from
        transfermarkt.passing_types
        where
        upper(position) not like '%GK%'
        )


        , goal_creation as 
        (
        SELECT
        season,
        player_club,
        player_name,
        SCA_Types_PassLive,
        SCA_Types_PassDead,
        GCA_Types_PassDead,
        SCA_SCA as SCA
        

        from
        transfermarkt.goals_created

        )

        ,
        join_pass_data as 
        (
        select
        a.*,
        SCA_Types_PassDead,
        GCA_Types_PassDead,
        SCA_Types_PassLive,
        b.SCA

        from
        set_piece_takers a

        left outer join 

        goal_creation b on 

        a.player_name = b.player_name and
        a.player_club = b.player_club and
        a.season = b.season
        )

        , metric_calculations as (
        select
        season,
        player_club,
        player_name,
        games_played,
        minutes_played,
        round(SAFE_DIVIDE(SCA_Types_PassLive,games_played),2) live_ball_scas_p90,
        round(SAFE_DIVIDE(pass_types_dead,games_played),2) dead_ball_taken_p90,
        round(SAFE_DIVIDE(SCA_Types_PassDead,games_played),2) dead_ball_scas_p90,
        round(SAFE_DIVIDE(GCA_Types_PassDead,games_played),2) dead_ball_gcas_p90,
        SAFE_DIVIDE(round(SAFE_DIVIDE(SCA_Types_PassDead,games_played),2),round(SAFE_DIVIDE(pass_types_dead,games_played),2)) dead_ball_sca_eff_p90,
        round(pass_types_dead/sum(pass_types_dead) over (partition by player_club,season),3) team_dead_ball_taken_pc,
        round(pass_types_ck/sum(pass_types_ck) over (partition by player_club,season),3) corner_taken_pc,
        round(pass_types_fk/sum(pass_types_fk) over (partition by player_club,season),3) free_kick_taken_pc,
        round(case when position = 'GK' then 0 else pass_types_dead end/sum(case when position = 'GK' then 0 else pass_types_dead end) over (partition by player_club,season),3) non_gk_dead_ball_pc,
        round(SCA_Types_PassDead/sum(SCA_Types_PassDead) over (partition by player_club,season),3) dead_ball_sc_pc,
        round(GCA_Types_PassDead/sum(GCA_Types_PassDead) over (partition by player_club,season),3) dead_ball_gc_pc,
        sum(SCA_Types_PassDead) over (partition by player_club, season) team_dead_ball_scas,
        sum(GCA_Types_PassDead) over (partition by player_club, season) team_dead_ball_gcas,
        round(sum(SCA_Types_PassDead) over (partition by player_club, season)/sum(sca) over (partition by player_club, season),3) team_dead_ball_sc_pc,
        case when round(SCA_Types_PassDead/sum(SCA_Types_PassDead) over (partition by player_club,season),3) > 0.1 then 'Starter' else 'Other' end taker_type


        from
        join_pass_data
        )


        select *, round(team_dead_ball_gcas/team_dead_ball_scas,3) dead_ball_success_rate from metric_calculations 
        where
        (
        dead_ball_sc_pc > 0.1
        or
        dead_ball_scas_p90 > 0.8
        )


        order by season desc, player_club asc, team_dead_ball_sc_pc desc, dead_ball_sc_pc desc
        """)

st.write(
"""
The key part of the criteria is defined in the final select statement. A player has to have either 
- Taken greater than 10% of all shot creating set pieces for the club in the season
OR
- Have a dead ball shot creation rate of greater than 0.8 per 90 mins (to account for substitute players)

This second rule has proven vital for fringe set-piece takers who then go on to be first team starters such as Michael Kayode for Brentford.

The hard part for the user is determining which proportion of the set-pieces will be taken by each player in the match. This can vary quite a lot depending
on who is on the pitch at the time and with some notable exceptions like Alexander-Arnold or Trippier, clubs rarely have one dedicated set-piece taker.

True value in this calculation will be unlocked from correctly predicting a player who doesn't usually take set-pieces being assigned the role for a game. I would like
to create a form of predictive model that given a starting lineup can distribute the set-piece proportions automatically.

Carrying on with the Crystal Palace vs Brighton example:
"""
)

with st.echo():
    set_piece_takers = pd.read_csv("data/data/fbref_dashboard/set_piece_takers_fbref.csv")
    
    clubs = ['Crystal Palace', 'Brighton']
    season = 2024

    example_set_piece_takers = set_piece_takers[
        (set_piece_takers["player_club"].isin(clubs)) &
        (set_piece_takers["season"] == season)
    ]
    
    st.dataframe(example_set_piece_takers)

st.write(
"""
and choosing Pervis Estupiñán and Eberechi Eze for simplicity, we can now predict goalscorer probabilities.
"""
)

st.subheader("Putting it all together")

st.write(
"""
Now we have all of the necessary inputs we can model the probability of our centre-backs scoring, assisted by our chosen set piece takers.
"""
)

with st.echo():
    import numpy as np 
    
    def cb_score_spt_assist(team_lambda, cb_goal_contr, spt_assist_contr = 0.47, spt_taker_prop = 0.8):
        pct_chance = 1-np.exp(-team_lambda*cb_goal_contr*spt_assist_contr*spt_taker_prop)
        return pct_chance

    eze_guehi_prob = cb_score_spt_assist(lambda_palace, home_defender_cont)
    pervis_dunk_pron = cb_score_spt_assist(lambda_brighton, away_defender_cont)
    
    st.write(f"The probability that {home_defender} will score a goal assisted by Eze is {round(eze_guehi_prob*100,3)}%")
    st.write(f"The probability that {away_defender} will score a goal assisted by Estupiñán is {round(pervis_dunk_pron*100,3)}%")
    
st.write(
"""
You'll notice that it doesn't actually matter which set piece taker is chosen by the user, only changing the proportion of set-pieces they take will affect the probability.

An improvement here would be to increase or decrease the 47% figure based on how efficient the set-piece taker is.
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
