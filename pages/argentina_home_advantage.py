import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import inspect
import seaborn as sns
from scipy.stats import poisson, skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf

st.title("Home Advantage in Argentinian Football")
st.caption("The following article is one of the first pieces of ad-hoc python analysis I did in 2023. I followed this [article](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/) substituting the data with my own use case.")
# Introduction
st.markdown("""
### Introduction
Recently I was fortunate enough to attend two matches at Boca Juniors' infamous Bombanera Stadium. The first being a rather forgettable (at least football-wise) 2-1 loss against bottom of the league Colón. The second was a more convincing 1-0 victory against Huracán.

In truth the real attraction to Argentian football, or South American football in general, is the atmosphere. The fans never stop chanting, singing and dancing regardless of what is occurring on the pitch. A far cry from English football where the crowd mood is entirely dictated by the flow of the game.

I was so taken by the atmosphere of the first game I attended, I didn't realise something was missing - away fans. This is because in 2013 the Argentinian FA chose to outlaw away fans after a spate of football-related violence. Home advantage is a well-documented phenomenon in football, [this was reaffirmed](https://www.sciencedirect.com/science/article/pii/S146902922100131X#:~:text=COVID%2D19%20provided%20an%20opportunity,fewer%20goals%2C%20with%20crowds%20absent.) during the COVID seasons in which home teams won fewer games than the long run average.

This got me to thinking; How much has this change in regulation changed home advantage in Argentina?
""")

st.header("Data Preparation", divider = True)

# File upload
fb_data = "data/data/arg_fb_data.csv"

def data_prep(file):
    # Import data and select relevant columns
    arg_league_data = pd.read_csv("data/data/arg_fb_data.csv")
    arg_league_data.columns.str.strip()
    arg_league_cropped = arg_league_data[['League','Season','Home','Away','HG','AG']]

    # Strip trailing space pre 2017/18 from league
    arg_league_cropped.League = arg_league_cropped['League'].str.strip()
    arg_league_cropped = arg_league_cropped[arg_league_cropped.League == "Liga Profesional"]

    # Split into two data sets pre and post ban
    arg_league_cropped = arg_league_cropped[arg_league_cropped.League == "Liga Profesional"]
    arg_league_pre = arg_league_cropped[arg_league_cropped.Season == "2012/2013"]
    arg_league_post = arg_league_cropped[arg_league_cropped.Season.isin(["2013/2014", "2014/2015"])]

    return arg_league_cropped, arg_league_pre, arg_league_post

st.code(inspect.getsource(data_prep))
arg_league_cropped, arg_league_pre, arg_league_post = data_prep(fb_data)


st.write("### Data Sample")
st.write(arg_league_cropped.head())



with st.echo():
    arg_league_goals_pre = arg_league_pre.iloc[:, -2:]
    pre_mean_goals = arg_league_goals_pre.mean()
    st.dataframe(pre_mean_goals)

with st.echo():
    arg_league_goals_post = arg_league_post.iloc[:, -2:]
    post_mean_goals = arg_league_post.iloc[:, -2:].mean()
    st.dataframe(post_mean_goals)
    
st.markdown("""
We can see in both time periods, the number of home goals exceeds the number of away goals.
This passes the initial assumption that home advantage exists to some degree. 
However, the average goals scored for away teams appears to decrease after the ban on away fans. 
Similarly, the number of goals scored by the home team decreases. This will be examined in more detail later.
""")

# Mean goals
st.write("#### Average Goals per Match")
st.write("Pre-Ban:", arg_league_pre.HG.mean())
st.write("Post-Ban:", arg_league_post.AG.mean())


st.header("Poisson Distribution", divider = True)
st.markdown(
"""
We can use a poisson distribution to calculate the expected number of goals for both the home and away team in any given match in this league. 
This is not in itself a particularly useful statistic for estimating scores of isolated matches as it negates the difference in quality of each team.
 Rather this can be used to assess the overall affect of home advantage
"""    
)


# Poisson Distribution

hg_prop = arg_league_post.groupby("HG").count()
hg_prop['Home Goal Proportion'] = hg_prop['League'] / hg_prop['League'].sum()
hg_prop = hg_prop[['Home Goal Proportion']]

ag_prop = arg_league_post.groupby("AG").count()
ag_prop['Away Goal Proportion'] = ag_prop['League'] / ag_prop['League'].sum()
ag_prop = ag_prop[['Away Goal Proportion']]


# Merge the two data sets
merged_prop = hg_prop.join(ag_prop, how='outer')
# Display merged dataframe
merged_prop.fillna(0, inplace=True)
# Convert index to new column called goals
merged_prop = merged_prop.reset_index().rename(columns={'index': 'Goals'})

# Display updated dataframe
st.dataframe(merged_prop)


def goal_prop_bar_chat():
    # Create catplot with goals as x-axis and HG Proportion and AG Proportion as y-values
    fig = sns.catplot(x='Goals',
                y='value',
                hue='variable',
                data=pd.melt(merged_prop, id_vars = 'Goals'),
                kind='bar',
                palette='pastel',
                legend_out=False)
    

    # Set chart title and axis labels
    plt.title('Home and Away Goal Proportions')
    plt.xlabel('Goals')
    plt.ylabel('Proportion')

    plt.legend(title='Proportion', loc='upper right')
    
    return fig

g1 = goal_prop_bar_chat()

st.pyplot(g1)

st.markdown(
r"""
with the actual number of goals scored in each game plotted, we can simulate the expected number of goals scored by fitting a poisson distribution to the data.
For reference, the formula for the poisson distribution is as follows: $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$ where lambda is the average number of successes (in our case goals) and k is the number of successes we want to predict a probability for
"""
)

def overlaid_poisson():
        # Create catplot with goals as x-axis and HG Proportion and AG Proportion as y-values
    b = sns.catplot(x='Goals',
                y='value',
                hue='variable',
                data=pd.melt(merged_prop, id_vars = 'Goals'),
                kind='bar',
                palette='pastel',
                legend_out=False)
   

    # Set chart title and axis labels
    plt.title('Home and Away Goal Proportions')
    plt.xlabel('Goals')
    plt.ylabel('Proportion')


    # Set lambda parameter
    lam_hg = 1.289557
    lam_ag = 0.973629

    # Create Poisson distribution
    poisson_dist_hg = poisson(lam_hg)
    poisson_dist_ag = poisson(lam_ag)

    # Plot probability mass function of Poisson distribution
    x = np.arange(0, 8)
    plt.plot(x, poisson_dist_hg.pmf(x), 'o-', color = 'blue', ms=8, label='Poisson Home Goal')
    plt.plot(x, poisson_dist_ag.pmf(x), 'o-', color = 'green', ms=8, label='Poisson Away Goal')

    plt.legend(title='Proportion/Prediction', loc='upper right')
    return b, poisson_dist_hg, poisson_dist_ag

g2, poisson_dist_hg, poisson_dist_ag = overlaid_poisson()

st.pyplot(g2)

st.markdown(
"""
Now we have the proportion of actual goals scored by both the home and away teams in the league versus a prediction from their corresponding Poisson probability mass functions. It can be seen that the fit for the home goals is far better than away, but both are fairly accurate.

We can also estimate the probability of a selected number of goals to be scored by each team. For example the probability of the home team scoring 3 goals in any argentinian league game according to the poisson distribution is
"""    
)

with st.echo():
    three_hg_prob = poisson_dist_hg.pmf(3)
    three_hg_prop = merged_prop.iloc[3,1]
    st.write((f"The probability of the home team scoring 3 goals is {round(three_hg_prob*100,2)}% whereas the proportion of goals scored is {round(three_hg_prop*100,2)}%"))

st.header("The Skellam Distribution")

st.markdown(
"""
A skellam distribution gives the probability of the difference of two independent random variables who are poisson distributed. An example would be calculating the goal difference of a home and away teams' goals based on the probability of the home team scoring not affecting the probability of the away team scoring).
"""
)

with st.echo():
    pre_draw_chance = skellam.pmf(0,arg_league_goals_pre.mean()[0],arg_league_goals_pre.mean()[1])
    post_draw_chance = skellam.pmf(0,arg_league_goals_post.mean()[0],arg_league_goals_post.mean()[1])
    st.write((f"The probability of a draw in the season pre-ban was {round(pre_draw_chance*100,2)}% whereas the probability of a draw post-ban is {round(post_draw_chance*100,2)}%"))
    
skellam_pred_pre = [skellam.pmf(i,  arg_league_goals_pre.mean()[0],  arg_league_goals_pre.mean()[1]) for i in range(-6,8)]
skellam_pred_post = [skellam.pmf(i,  arg_league_goals_post.mean()[0],  arg_league_goals_post.mean()[1]) for i in range(-6,8)]


def skellam_plots():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(arg_league_pre[['HG']].values - arg_league_pre[['AG']].values, range(-6,8), 
            alpha=0.7, label='Actual', density = True)
    axs[0].plot([i+0.5 for i in range(-6,8)], skellam_pred_pre,
                    linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')
    axs[0].legend(loc='upper right', fontsize=13)
    axs[0].set_xticks([i+0.5 for i in range(-6,8)])
    axs[0].set_xticklabels([i for i in range(-6,8)])
    axs[0].set_xlabel("Home Goals - Away Goals",size=13)
    axs[0].set_ylabel("Proportion of Matches",size=13)
    axs[0].set_title(f"Difference in Goals Scored (Home Team vs Away Team) Pre-Ban, n={arg_league_pre.Season.count()}",size=8,fontweight='bold')
    axs[0].set_ylim([-0.004, 0.35])

    axs[1].hist(arg_league_post[['HG']].values - arg_league_post[['AG']].values, range(-6,8), 
            alpha=0.7, label='Actual', density = True)
    axs[1].plot([i+0.5 for i in range(-6,8)], skellam_pred_post,
                    linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')
    axs[1].legend(loc='upper right', fontsize=13)
    axs[1].set_xticks([i+0.5 for i in range(-6,8)])
    axs[1].set_xticklabels([i for i in range(-6,8)])
    axs[1].set_xlabel("Home Goals - Away Goals",size=13)
    axs[1].set_ylabel("Proportion of Matches",size=13)
    axs[1].set_title(f"Difference in Goals Scored (Home Team vs Away Team) Post-Ban, n={arg_league_post.Season.count()}",size=8,fontweight='bold')
    axs[1].set_ylim([-0.004, 0.35])

    return fig

skellam = skellam_plots()


st.pyplot(skellam)
with st.expander("See Code"):
    st.code(inspect.getsource(skellam_plots))

st.write("using this data we can build a poisson regression model to predict the scores of each game in the league.")

st.header("Regression Model")

with st.echo():
    goal_model_data_pre = pd.concat([arg_league_pre[['Home','Away','HG']].assign(home=1).rename(
            columns={'Home':'team', 'Away':'opponent','HG':'goals'}),
           arg_league_pre[['Away','Home','AG']].assign(home=0).rename(
            columns={'Away':'team', 'Home':'opponent','AG':'goals'})])

    goal_model_data_post = pd.concat([arg_league_post[['Home','Away','HG']].assign(home=1).rename(
            columns={'Home':'team', 'Away':'opponent','HG':'goals'}),
           arg_league_post[['Away','Home','AG']].assign(home=0).rename(
            columns={'Away':'team', 'Home':'opponent','AG':'goals'})])
    
    poisson_model = smf.glm(formula="goals ~ home + team + opponent", data = goal_model_data_pre,
                        family = sm.families.Poisson()).fit()
    poisson_model_p = smf.glm(formula="goals ~ home + team + opponent", data = goal_model_data_post,
                        family = sm.families.Poisson()).fit()
    st.write(poisson_model.summary())

st.write(
"""
From this model we can create a function to simulate fixtures of our choosing.
"""
)

with st.echo():
    def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
        home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                                'opponent': awayTeam,'home':1},
                                                        index=[1])).values[0]
        away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                                'opponent': homeTeam,'home':0},
                                                        index=[1])).values[0]
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
        return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

st.write(
"""
Now we just choose our home team and away team and call the function which will return a matrix of all possible goal combinations (up to our max goal parameter).
"""    
)

with st.echo():
    river_belg_post_ban = simulate_match(poisson_model_p,"River Plate","Belgrano",max_goals=10)
    st.write(river_belg_post_ban)
    
st.write(
"""
Then to calculate the probability of the match winner we can sum the probabilities under the diagonal for a home win, above the diagonal for away and through the diagonal for a draw.
"""    
)

# Convert NumPy array to Pandas DataFrame (Add column/row labels if needed)
df = pd.DataFrame(river_belg_post_ban)  # Convert your array

# Function to highlight lower triangle
def highlight_lower_triangle(data):
    """Returns a DataFrame with styles for the lower triangle."""
    styles = pd.DataFrame("", index=data.index, columns=data.columns)  # Create empty style DataFrame
    for i in range(len(data)):
        for j in range(len(data.columns)):
            if i > j:  # Only color below diagonal
                styles.iloc[i, j] = "background-color: lightgreen"
            elif i == j:
                styles.iloc[i, j] = "background-color: orange"
            else:
                styles.iloc[i, j] = "background-color: red"
    return styles

# Apply styling to DataFrame
styled_df = df.style.apply(highlight_lower_triangle, axis=None)

# Display in Streamlit
st.write(styled_df)


with st.echo():
    river_win = np.sum(np.tril(river_belg_post_ban,-1))
    draw = np.sum(np.diag(river_belg_post_ban,0))
    belg_win = np.sum(np.triu(river_belg_post_ban,1))

    st.write(f"The probability of River Plate beating Belgrano at home  post away fans ban is {river_win:.1%}")
    st.write(f"The probability of a draw between the two teams post ban is {draw:.1%}")
    st.write(f"The probability of Belgrano beating River Plate away from home post away fans ban is {belg_win:.1%}")
    
st.markdown(
"""
if we want to generalise the impact of how much more likely a home team is to score now compared to when away fans were allowed
we simply need to subtract the home coefficients away from one another and raise against the natural exponent.
"""    
)

with st.echo():
    st.write(f"Prior to away fans being banned, the home team were {round(np.exp(poisson_model.params.home),2)} times more likely to score than the away team. Post-ban this changes to {round(np.exp(poisson_model_p.params.home),2)}")

st.markdown(
"""
This would suggest that the home advantage did indeed increase in the two seasons following the banning of away fans.
"""    
)