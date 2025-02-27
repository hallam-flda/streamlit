# Import necessary packages

from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns
import pandas as pd
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go
import pickle

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

st.title("Simulations")

st.markdown(
"""
Due to the size of the output files I will only be sharing the results as plots and graphs but I will include all code for the 
sake of reproducibility.
"""
)

st.header("Base Simulation", divider = True)

st.markdown(
"""
First it makes sense to plot the results of a simulation with the base assumption of a £100 budget and a maximum spin allowance of 20,000
"""
)

with st.expander("See Code"):
    st.code(
        """
        from dataclasses import dataclass
import numpy as np
import pickle

np.random.seed(17)


@dataclass
class SimulationResult:
    spins: int
    balance: int
    balance_history: list
    casino_profit: list
    stop_reason: str


# Constants for different games
GAME_PARAMS = {
    'colours': {
        'win_prob': 0.48649,
        'win_payout': 1,
        'centre': -1 / 37,
        'sigma': 6 * np.sqrt(38) / 37,
        'loss_streak': 17
    },
    'numbers': {
        'win_prob': 1 / 37,
        'win_payout': 35,
        'centre': -1 / 37,
        'sigma': 5.84,
        'loss_streak': 290
    }
}


def unlucky_balance_condition(balance_history, game='colours'):
    '''Check if the player's balance is significantly below expectations.'''
    if len(balance_history) < 100:
        return False
    params = GAME_PARAMS[game]
    t = len(balance_history)
    center = params['centre'] * t
    std_dev = params['sigma'] * np.sqrt(t)
    lower_threshold = 100 + (center - 1.645 * std_dev)
    return balance_history[-1] < lower_threshold


def european_roulette_simulation(max_spins: int, starting_balance: int, game='colours') -> SimulationResult:
    spins = 0
    balance = starting_balance
    balance_history = [balance]
    casino_profit = [0]
    consecutive_losses = 0
    stop_reason = None

    params = GAME_PARAMS[game]

    while True:
        if balance <= 0:
            stop_reason = 'zero_balance'
            break
        if spins >= max_spins:
            stop_reason = 'max_spins'
            break
        # if consecutive_losses >= params['loss_streak']:
        #     stop_reason = 'loss_streak'
        #     break
        # if unlucky_balance_condition(balance_history, game):
        #     stop_reason = 'unlucky_dist'
        #     break

        # Simulate a spin outcome
        outcome = params['win_payout'] if np.random.rand() < params['win_prob'] else -1
        balance += outcome
        spins += 1
        balance_history.append(balance)
        casino_profit.append(starting_balance - balance)
        consecutive_losses = consecutive_losses + 1 if outcome == -1 else 0

    return SimulationResult(spins, balance, balance_history, casino_profit, stop_reason)


def run_simulations(n=10000, max_spins=20000, starting_balance=100, game='colours'):
    results = []
    for i in range(n):
        result = european_roulette_simulation(max_spins, starting_balance, game)
        results.append(result)
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"Completed {i + 1}/{n} simulations for game: {game}")
    return results


def compute_rolling_margin(casino_profit_lists):
    '''
    Compute a "rolling margin" ratio.

    Given a list of casino_profit lists (each list corresponding to one simulation),
    this function creates a padded 2D array (using np.nan for missing values), then:
      - For column 0, uses the sum of valid entries.
      - For each subsequent column, adds the sum of differences (current - previous)
        for rows that have valid values.
      - Finally, the ratio is cumulative difference divided by the cumulative count
        of non-NaN values.
    '''
    # Determine maximum length
    max_len = max(len(lst) for lst in casino_profit_lists)
    matrix = np.full((len(casino_profit_lists), max_len), np.nan)
    for i, lst in enumerate(casino_profit_lists):
        matrix[i, :len(lst)] = lst

    non_nan_counts = np.sum(~np.isnan(matrix), axis=0)
    cumulative_counts = np.cumsum(non_nan_counts)

    cumulative_diff = np.empty(max_len)
    cumulative_diff[0] = np.nansum(matrix[:, 0])
    for j in range(1, max_len):
        valid = ~np.isnan(matrix[:, j]) & ~np.isnan(matrix[:, j - 1])
        diff_sum = np.sum(matrix[valid, j] - matrix[valid, j - 1])
        cumulative_diff[j] = cumulative_diff[j - 1] + diff_sum

    final_ratio = cumulative_diff / cumulative_counts
    return final_ratio


def aggregate_results(results):
    '''
    Given a list of SimulationResult objects, extract:
      - overall rolling margin (from casino_profit, excluding the first value),
      - a list of spins,
      - closing balances,
      - total spins,
      - and grouping by stop_reason for individual rolling margins.
    Returns a dictionary of aggregated stats.
    '''
    # Extract overall lists from all simulations
    casino_profit_lists = [r.casino_profit[1:] for r in results]  # exclude initial value
    overall_rolling_margin = compute_rolling_margin(casino_profit_lists)
    spin_list = [r.spins for r in results]
    closing_balances = [r.balance_history[-1] for r in results]
    total_spins = sum(spin_list)
    stop_reasons = [r.stop_reason for r in results]

    # Group simulations by stop_reason
    grouped = {}
    for r in results:
        grouped.setdefault(r.stop_reason, []).append(r)

    grouped_rolling_margin = {}
    for stop_reason, group in grouped.items():
        cp_lists = [r.casino_profit[1:] for r in group]
        if cp_lists:
            grouped_rolling_margin[stop_reason] = compute_rolling_margin(cp_lists)
        else:
            grouped_rolling_margin[stop_reason] = None

    return {
        "overall": {
            "rolling_margin": overall_rolling_margin,
            "spin_list": spin_list,
            "closing_balances": closing_balances,
            "total_spins": total_spins,
            "stop_reasons": stop_reasons
        },
        "by_stop_reason": grouped_rolling_margin
    }


if __name__ == '__main__':
    # Run simulations for both games
    simulation_output = {}
    for game in GAME_PARAMS.keys():
        print(f"\nRunning simulations for game: {game}")
        sim_results = run_simulations(n=10000, max_spins=20000, starting_balance=100, game=game)
        simulation_output[game] = aggregate_results(sim_results)

    # Save the aggregated simulation output into a pickle file.
    with open('simulation_results.pickle', 'wb') as f:
        pickle.dump(simulation_output, f)

    print("\nSimulation results have been saved to 'simulation_results.pickle'.")
    
        """)

loading_pickle = """
with open('data/data/projects/roulette_sims/simulation_results_no_conditions.pickle', 'rb') as f:
    results = pickle.load(f)

st.write(results)
"""

with st.expander("See Pickle Loading and Data"):
    st.code(loading_pickle)
    exec(loading_pickle)

colours_rolling_margin = results["colours"]["overall"]["rolling_margin"]

st.subheader("Margin Performance")


# Create three tabs
tab1, tab2, tab3 = st.tabs(["Short term", "Medium term", "Long term"])

# --- Short term tab ---
with tab1:
    st.subheader("Short Term (0–100 spins)")
    st.write("Too much noise early on without any chance for customers to lapse (at least 100 spins required)")
    fig, ax = plt.subplots()
    ax.plot(colours_rolling_margin)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(0, 100)
    ax.set_ylim(0.02, 0.05)
    st.pyplot(fig)

# --- Medium term tab ---
with tab2:
    st.subheader("Medium Term (100–4000 spins)")
    st.write("In the medium term we observe marginally above average profit margins for the casino")
    fig, ax = plt.subplots()
    ax.plot(colours_rolling_margin)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(100, 4000)
    ax.set_ylim(0.025, 0.03)
    st.pyplot(fig)

# --- Long term tab ---
with tab3:
    st.subheader("Long Term (4000–20000 spins)")
    st.write("In the long term the excess profit persists but is diluted by players on long streaks")
    fig, ax = plt.subplots()
    ax.plot(colours_rolling_margin)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(4000, 20000)
    ax.set_ylim(0.026, 0.028)
    st.pyplot(fig)

st.subheader("Spin Distribution")
st.write(
"""
The distribution of spins is skewed due to the budget constraint we introduced, however, we can see that the 
average number of spins matches the theory of gambler's ruin with a £100 starting balance.
"""
)
fig, ax = plt.subplots()
plt.hist(results["colours"]["overall"]["spin_list"], bins = 50)
ax.axvline(x=3703, linestyle = '--', color = 'green')

annotation = f'''
Theoretical average spins to bankruptcy from £100 = 3,703 \n
Mean spins to bankruptcy in sample = {np.mean(results["colours"]["overall"]["spin_list"]):.2f}
'''

ax.text(
    5000,        # x in data coordinates
    800,          # y in data coordinates (pick a suitable y that is inside your plot area)
    annotation, 
    color='black',
    fontsize=8
)

st.pyplot(fig)

with st.expander("Why is this not normal?"):
    st.write(
    """
    The distribution of spins has a left tail due to the budget constraint introduced. When approximating the normal distribution
    there is no condition of budget, meaning that players can come back from a losing position of more than £100. In reality
    this does not work because we assume players cannot play with negative balance.
    """)

st.subheader("Stopping Reason")
st.write(
"""
Finally we can look at the reason for customers stopping. In the base case we expect this to be primarily due to running out of budget
"""
)

df_stop_reason = pd.DataFrame(results["colours"]["overall"]["stop_reasons"])
df_stop_reason = df_stop_reason.value_counts()
st.dataframe(df_stop_reason)


###
### ------------------- WITH CONSTRAINTS ---------------
###

st.header("With Luck Conditions", divider = True)

loading_pickle_cond = """
with open('data/data/projects/roulette_sims/simulation_results.pickle', 'rb') as f:
    results_cond = pickle.load(f)

st.write(results_cond)
"""

with st.expander("See Pickle Loading and Data"):
    st.code(loading_pickle_cond)
    exec(loading_pickle_cond)

colours_rolling_margin_cond = results_cond["colours"]["overall"]["rolling_margin"]

st.subheader("Margin Performance")


# Create three tabs
tab1, tab2, tab3 = st.tabs(["Short term", "Medium term", "Long term"])

# --- Short term tab ---
with tab1:
    st.subheader("Short Term (0–100 spins)")
    st.write("Too much noise early on without any chance for customers to lapse (at least 100 spins required)")
    fig, ax = plt.subplots()
    ax.plot(colours_rolling_margin_cond)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(0, 100)
    ax.set_ylim(0.02, 0.05)
    st.pyplot(fig)

# --- Medium term tab ---
with tab2:
    st.subheader("Medium Term (100–4000 spins)")
    st.write("In the medium term we observe marginally above average profit margins for the casino")
    fig, ax = plt.subplots()
    ax.plot(colours_rolling_margin_cond)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(100, 4000)
    ax.set_ylim(0.025, 0.03)
    st.pyplot(fig)

# --- Long term tab ---
with tab3:
    st.subheader("Long Term (4000–20000 spins)")
    st.write("In the long term the excess profit persists but is diluted by players on long streaks")
    fig, ax = plt.subplots()
    ax.plot(colours_rolling_margin_cond)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(4000, 20000)
    ax.set_ylim(0.026, 0.028)
    st.pyplot(fig)

st.subheader("Spin Distribution")
st.write(
"""
The distribution of spins is skewed due to the budget constraint we introduced, however, we can see that the 
average number of spins matches the theory of gambler's ruin with a £100 starting balance.
"""
)
fig, ax = plt.subplots()
plt.hist(results_cond["colours"]["overall"]["spin_list"], bins = 50)
ax.axvline(x=3703, linestyle = '--', color = 'green')

annotation = f'''
Theoretical average spins to bankruptcy from £100 = 3,703 \n
Mean spins to bankruptcy in sample = {np.mean(results_cond["colours"]["overall"]["spin_list"]):.2f}
'''

ax.text(
    5000,        # x in data coordinates
    800,          # y in data coordinates (pick a suitable y that is inside your plot area)
    annotation, 
    color='black',
    fontsize=8
)

st.pyplot(fig)

st.subheader("Stopping Reason")
st.write(
"""
Finally we can look at the reason for customers stopping. In the base case we expect this to be primarily due to running out of budget
"""
)

df_stop_reason = pd.DataFrame(results_cond["colours"]["overall"]["stop_reasons"])
df_stop_reason = df_stop_reason.value_counts()
st.dataframe(df_stop_reason)


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

with tab_numbers:

###
### -------------- NUMBERS -------------
###

numbers_rolling_margin = results["numbers"]["overall"]["rolling_margin"]

st.subheader("Margin Performance")


# Create three tabs
tab1, tab2, tab3 = st.tabs(["Short term", "Medium term", "Long term"])

# --- Short term tab ---
with tab1:
    st.subheader("Short Term (0–100 spins)")
    st.write("Too much noise early on without any chance for customers to lapse (at least 100 spins required)")
    fig, ax = plt.subplots()
    ax.plot(numbers_rolling_margin)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.05)
    st.pyplot(fig)

# --- Medium term tab ---
with tab2:
    st.subheader("Medium Term (100–4000 spins)")
    st.write("In the medium term we observe marginally above average profit margins for the casino")
    fig, ax = plt.subplots()
    ax.plot(numbers_rolling_margin)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(100, 4000)
    ax.set_ylim(0.02, 0.035)
    st.pyplot(fig)

# --- Long term tab ---
with tab3:
    st.subheader("Long Term (4000–20000 spins)")
    st.write("In the long term the excess profit persists but is diluted by players on long streaks")
    fig, ax = plt.subplots()
    ax.plot(numbers_rolling_margin)
    ax.axhline(y=0.027027, linestyle='--', color='red')
    ax.set_xlim(4000, 100000)
    ax.set_ylim(0.025, 0.03)
    st.pyplot(fig)

st.subheader("Spin Distribution")
st.write(
"""
The distribution of spins is skewed due to the budget constraint we introduced, however, we can see that the 
average number of spins matches the theory of gambler's ruin with a £100 starting balance.
"""
)
fig, ax = plt.subplots()
plt.hist(results["numbers"]["overall"]["spin_list"], bins = 50)
ax.axvline(x=3703, linestyle = '--', color = 'green')

annotation = f'''
Theoretical average spins to bankruptcy from £100 = 3,703 \n
Mean spins to bankruptcy in sample = {np.mean(results["numbers"]["overall"]["spin_list"]):.2f}
'''

ax.text(
    5000,        # x in data coordinates
    800,          # y in data coordinates (pick a suitable y that is inside your plot area)
    annotation, 
    color='black',
    fontsize=8
)

st.pyplot(fig)

with st.expander("Why is this not normal?"):
    st.write(
    """
    The distribution of spins has a left tail due to the budget constraint introduced. When approximating the normal distribution
    there is no condition of budget, meaning that players can come back from a losing position of more than £100. In reality
    this does not work because we assume players cannot play with negative balance.
    """)

st.subheader("Stopping Reason")
st.write(
"""
Finally we can look at the reason for customers stopping. In the base case we expect this to be primarily due to running out of budget
"""
)

df_stop_reason = pd.DataFrame(results["numbers"]["overall"]["stop_reasons"])
df_stop_reason = df_stop_reason.value_counts()
st.dataframe(df_stop_reason)