# Import necessary packages


from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns
import pandas as pd
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go

st.title("European Roulette Simulations")

st.header("Introduction", divider = True)
st.write("""
I want to look at what happens when introducing player behaviour on the return to player (RTP) of roulette. 
         Typically RTPs are measured over a large sample of plays but they ignore the fact that most customers are
         constrained by some limit of budget or time. I want to measure how much of an effect this could have on the 
         RTP to the customer and equally the profit margin recorded by the casino.
            """)

st.subheader("The Constraints")
st.markdown(
"""
The constraints are intended to act as a proxy for real-life behaviour. I want to see if when a player starts to feel 
'unlucky' what the effect is of losing this player on the total profit for the casino.

To keep it simple I will start with the following criteria:

<ul> 
<li> Customers all have a £100 starting balance
<li> Customers all place a £1 stake every turn
<li> Customers only choose red or black
<li> The maximum number of spins a customer can have is 20,000
</ul>

**Additional to this criteria there are two that will act as indicators of 'luck'.**
            
<ul>
<li> A customer will stop transacting after successive losses.
<li> A customer will stop transacting when in a cumulatively 'unlucky' position.
</ul>

Both of these are rather ambiguous so I will define them in some more detail.
            
""",
unsafe_allow_html=True)

st.subheader("Successive Losses")
st.markdown(
r"""
The idea is to create a criteria in which a customer decides that they have lost too many games in a row and would
like to stop losing.

That begs the question, how many games lost is too many? Form the theory work on Stochastic Processes, the expected number of spins
for a £100 starting budget with £1 spins on European Roulette colour betting is 3,703 spins.

One way to calculate an 'unlucky' streak in the context of this simulation is to visualise how frequently losing
streaks occur and pick a threshold based on the answer.

The formula to calculate this is given by the equation:

$$
P = 1 - \exp\left(-3703 \cdot \left(P_{loss}\right)^{s} \right)
$$

where $P_{loss}$ is the probability of losing per spin and $s$ is the length of the losing streak
""", unsafe_allow_html=True)

successive_loss_code = """
# Define parameters
loss_prob = 19 / 37  

def estimate_run_probability_adjusted(flips, streak_length, loss_prob):
    if flips < streak_length:
        return 0  # Impossible to have the streak
    p_streak = (loss_prob) ** streak_length
    return 1 - np.exp(-flips * p_streak)

expected_games_colours = 3703

avg_colour_game_dict = {
    "Probability": [estimate_run_probability_adjusted(expected_games_colours, streak, loss_prob) for streak in range(1,20)]    

}

df = pd.DataFrame(avg_colour_game_dict, index = range(1,20))
df.index.name = "Streak Length"

fig, ax = plt.subplots()
plt.bar(df.index, df.Probability)
plt.title("Probability of Losing Streak in 3,703 Spins")
st.pyplot(fig)

"""
with st.expander("See Code"):
    st.code(successive_loss_code)

exec(successive_loss_code)


st.markdown(
"""
Personally I would be quite upset after only 10 successive losses, however, the maths tells us that there is a
99.1% chance this will happen when playing 3,703 spins of roulette. For the sake of the analysis, I will choose a threshold 
that is actually considered reasonably unlucky.

There is only a 4.3% chance that you will experience a loss streak of 17 in 3,703 so I will use this as the 'unlucky successive losses'
parameter to begin with.
"""
)

st.subheader("Cumulative Unlucky Position")

st.markdown(
"""
I want to also capture the scenario in which a customer is unlucky but doesn't experience a drastic loss streak. For example,
if for every 15 spins they win 1 time but lose 14 times, while they are not triggering the successive losses criteria they are still
experiencing an abnormal level of losses.

To capture this behaviour I want to flag anybody who is in the bottom 5% of the normal distribution generated in the 
first Stochastic Processes article.


"""    )

unlucky_distribution_graph = """

# Slider for the number of spins (t)
t = st.slider("Number of Spins (t)", min_value=100, max_value=2000, value=1400, step=100)

# Calculate distribution parameters based on t
sigma = 6 * np.sqrt(38) / 37       # per-bet sigma
center = -1/37 * t                 # drift (mean)
std_dev = sigma * np.sqrt(t)       # standard deviation

# Define the x-axis range as ±3 standard deviations around the center
x_min = center - 3 * std_dev
x_max = center + 3 * std_dev
lower_bound = center - 1.645 * std_dev

# Compute the PDF using scipy.stats.norm
x_values = np.linspace(x_min, x_max, 1000)
y_values = norm.pdf(x_values, loc=center, scale=std_dev)

# Create the Plotly figure
fig = go.Figure()

# Add the PDF curve
fig.add_trace(go.Scatter(
    x=x_values, 
    y=y_values,
    mode='lines',
    name='Normal PDF',
    line=dict(color='blue')
))

# Add vertical dashed lines for the lower bound
fig.add_shape(
    type="line", x0=lower_bound, x1=lower_bound, y0=0, y1=max(y_values),
    line=dict(color="red", dash="dash")
)

# Shade the left tail (x < lower_bound)
mask_left = x_values < lower_bound
fig.add_trace(go.Scatter(
    x=x_values[mask_left],
    y=y_values[mask_left],
    mode='lines',
    fill='tozeroy',
    fillcolor='rgba(173,216,230,0.5)',  # light blue fill
    line=dict(color='lightblue'),
    showlegend=False
))

# Update layout of the plot
fig.update_layout(
    title=f"PDF for European Roulette Colour Betting (t = {t})",
    xaxis_title="Balance",
    yaxis_title="Probability Density",
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
"""

with st.expander("See Code"):
    st.code(unlucky_distribution_graph)

exec(unlucky_distribution_graph)

st.markdown(
"""
Past a certain point, this threshold will not really apply as in order to be in the bottom 5% of the distribution after 1,400
spins you need to have lost more than £100 which is the pre-determined budget. I may need to adapt this and use a rolling window of 1,000
spins but I'll keep as-is for now.         
""")

st.header("Simulations", divider = True)

st.markdown(
"""
Due to the size of the output files I will only be sharing the results as plots and graphs but I will include all code for the 
sake of reproducibility.
"""
)

st.subheader("Without Luck Constraints")

st.markdown(
"""
First it makes sense to plot the results of a simulation with the base assumption of a £100 budget and a maximum spin allowance of 20,000
"""
)


# # Compute probabilities using the mathematical approximation for the new loss probability
# streaks = range(5, 21)
# flip_counts = range(500, 10500, 500)
# data_adjusted = {
#     flips: [estimate_run_probability_adjusted(flips, streak, loss_prob) for streak in streaks] for flips in flip_counts
# }

# # Create DataFrame
# df_adjusted = pd.DataFrame(data_adjusted, index=streaks)
# df_adjusted.index.name = "Streak Length"
# df_adjusted.columns.name = "Number of Flips"

# st.dataframe(df_adjusted)
