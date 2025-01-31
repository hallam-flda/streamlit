import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random



def european_roulette(max_number_of_spins, starting_balance, bet_size = 1, betting = "colours"):
    balance = []
    t = [0]
    running_balance = starting_balance
    running_t = 0
    
    if betting == "colours":
        for _ in range(max_number_of_spins):
            if running_balance <= 0:  # Stop if bankrupt
                break
            outcome = random.randint(0, 36)
            player_choice = random.randint(0,36)
            
            if outcome <= 17 :  # Winning outcomes
                running_balance += 1*bet_size  # Win pays 1 unit
            else:
                running_balance -= 1*bet_size  # Lose 1 unit
            
            # Track balance and time
            balance.append(running_balance)
            running_t += 1
            t.append(running_t)
    else:
        for _ in range(max_number_of_spins):
            if running_balance <= 0:  # Stop if bankrupt
                break
            outcome = random.randint(0, 36)
            player_choice = random.randint(0,36)
            
            if outcome == player_choice :  # Winning outcomes
                running_balance += 35*bet_size  # Win pays 35 units
            else:
                running_balance -= 1*bet_size  # Lose 1 unit
            
            # Track balance and time
            balance.append(running_balance)
            running_t += 1
            t.append(running_t)        
    
    spins_to_bankruptcy = running_t if running_balance <= 0 else max_number_of_spins  # None if not bankrupt
    return t, balance, spins_to_bankruptcy

# Set up the Streamlit application
st.title("European Roulette Colour Betting Simulation")

st.markdown(r"""
            

### 1.1 Bankruptcy Colours Betting

The obvious example we can use this theory for is the case in which the gambler loses all of their initial bankroll.

We will start by setting the customer's balance at £100 or $X_0 = 100$, what we're looking to find is where $\mathbb{E}[X_{t}] = 0$. We know that the absolute minimum time this could happen for £1 stakes would be 100 spins but winnings could greatly increase the time in which we expected to achieve a balane of zero. What we're looking for is the expected value of the stopping time,  $\mathbb{E}[\tau]$

As we've seen before this can be written as a random walk like so:


$$ 
X_{t+1} =  \begin{cases}
X_{t} + b, & \text{with probability } p \\
X_{t} - b, & \text{with probability } 1-p
\end{cases}
$$

where:


* $p$ is the probability of success, in this case correctly identifying a number
* $b$ is the bet size (and also the return size in the case of betting on colours)
* $X_t$ is the gambler's balance at time $t$




**Expected Change Per Step (Drift)**

The drift $ \Delta $ is the expected change in the gambler's bankroll after one step:



$$ \Delta = \mathbb{E}[X_{t+1} - X_{t}] = b \cdot p - b \cdot (1 - p). = bp - b + bp $$



$$
= 2bp - b = b(2p-1)
$$


The formula for the expected number of steps before bankruptcy is given by



$$
\mathbb{E}[\tau] = \frac{X_0}{|b\cdot(2p-1)|}
$$



which for £1 stakes, a starting balance of £100 and a win probability of $\frac{18}{37}$ gives us



$$
\mathbb{E}[\tau] = \frac{X_0}{b\cdot(1-2p)} = \frac{100}{1\cdot\left(1-2\cdot\left(\frac{18}{37}\right)\right)} = \frac{100}{0.027} \approx 3703
$$
            """)

st.markdown(r"""
## **Colours Betting**

Take the basic example of betting on colours with a starting balance of $X_0$ and £1 spins, how long will it take for the customer to reach zero on average?

We know from the Stochastic Processes work that for any value of $X_0$


$$
\mathbb{E}[\tau] = \frac{X_0}{b\cdot(1-2p)} = \frac{X_0}{1\cdot\left(1-2\cdot\left(\frac{18}{37}\right)\right)} = \frac{X_0}{0.027}
$$


""")

bet_per_spin = st.slider(r"Select Bet Per Spin, $b$:", min_value = 1, max_value = 10, step=1)
starting_balance = st.slider(r"Select Starting Balance, $X_{0}$:", min_value=20, max_value=200, step=10)
sims = st.slider(r"Select Number of Simulations", min_value = 100, max_value = 1000, step = 100)

# Run simulations
spins_to_bankruptcy = []
num_simulations = sims
for _ in range(num_simulations):
    _, _, spins = european_roulette(100000, starting_balance, bet_size = bet_per_spin, betting = "colours")
    spins_to_bankruptcy.append(spins)

mean_spins = np.mean(spins_to_bankruptcy)
st_dev_spins = np.std(spins_to_bankruptcy, ddof=1)

# Compute the value for the expected stopping time based on the formula
p = 18/37
b = bet_per_spin
expected_tau = starting_balance / (b * (1 - 2 * p))
e_t = round(expected_tau,1)

# Format the markdown string with the computed value
markdown_text = f"""
$$
\\mathbb{{E}}[\\tau] = \\frac{{X_0}}{{b\\cdot(1-2p)}} = \\frac{{X_0}}{{{bet_per_spin}\\cdot\\left(1-2\\cdot\\left(\\frac{{18}}{{37}}\\right)\\right)}} = \\frac{{{starting_balance}}}{{{bet_per_spin} \\cdot 0.027}} = {e_t} \\text{ spins}
$$
"""
st.markdown(markdown_text)

# st.write(f"Standard Deviation of Spins to Bankruptcy: {st_dev_spins:.0f}")

# Display results
fig, ax = plt.subplots()
ax.hist(spins_to_bankruptcy, bins=30, color='blue', alpha=0.7)
ax.set_title(f"Simulation of spins to bankruptcy for starting balance £{starting_balance}")
ax.set_xlabel("Number of Spins")
ax.set_ylabel("Frequency")

# Define the text for the textbox
stats_text = (f"Sample Mean = {mean_spins:.2f}\n"
              f"Expected Spins = {expected_tau:.1f}\n"
              f"Difference = {(expected_tau - mean_spins):.1f}")

# Place a text box in upper right in axes coords
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

st.pyplot(fig)



# Use st.markdown to render the formatted string


