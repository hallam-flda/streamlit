
import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

plt.rcParams['text.usetex'] = False


def random_walk(n_trials):
    balance = [0]  # Start at zero
    running_balance = 0
    for _ in range(n_trials):
        outcome = random.randint(0, 1)
        running_balance += 1 if outcome == 1 else -1
        balance.append(running_balance)
    return list(range(len(balance))), balance


def large_sim():
    fig, ax = plt.subplots()
    large_sim_x, large_sim_y = random_walk(1000)
    large_sim_x2, large_sim_y2 = random_walk(1000)
    large_sim_x3, large_sim_y3 = random_walk(1000)
    large_sim_x4, large_sim_y4 = random_walk(1000)
    large_sim_x5, large_sim_y5 = random_walk(1000)
    plt.plot(large_sim_x, large_sim_y)
    plt.plot(large_sim_x2, large_sim_y2)
    plt.plot(large_sim_x3, large_sim_y3)
    plt.plot(large_sim_x4, large_sim_y4)
    plt.plot(large_sim_x5, large_sim_y5)
    ax = plt.gca()  # Get current axes
    ax.spines['bottom'].set_position(('data', 0))
    plt.hlines(y=0, xmin=0,  xmax=len(large_sim_x),
               color='black', linestyle='--')
    plt.box(False)
    plt.show()
    return fig


st.title("""
**Stochastic Processes Theory - Stopping Times**
""")

st.header("""
Introduction
""")

st.markdown(r"""
In the previous notebook, we looked at modelling the European Roulette wheel as a stochastic process and analysed some of the properties of the expected distribution of final player balances.

Rather than depending on a few simulated outcomes, we can instead look to sample the underlying distributions that describe the data and draw on more Stochastic Process Theory to study how a stop-gap would affect the casino's profitability.
""")

st.header("""
          Stopping Times
""", divider=True)

st.subheader("""
          Mathematical Properties
""")

st.markdown(r"""
## Mathematical Properties
**Definition:** A stopping time $\tau$ is a random variable that represents the time at which a particular condition within a stochastic process is satisfied. The key property of a stopping time is that the decision to stop can be made based only on the information available up to that time.

**Formally:** $\tau$ is a stopping time with respect to a filtration: $\{{F}_t\}$ if $\{\tau \leq t\} \in F_t$ for all $t$

**What is a filtration?**

A filtration as denoted by $\{F_{t}\}$ is the set of all information up to time $t$. As time increases so does the amount of information stored in the filtration.

#### **Example: Filtration in Roulette**

- **$t = 0$**: You know the initial bankroll:  
  $F_0 = \{\text{Bankroll: 1000}\}$

- **$t = 1$**: The first spin is red, you bet £10 and win:  
  $F_1 = \{\text{Bankroll: 1010, First Spin: Red}\}$

- **$t = 2$**: The second spin is black, you bet £20 and lose:  
  $F_2 = \{\text{Bankroll: 990, Spins: [Red, Black]}\}$

- **$t = 3$**: The third spin is red, you bet £30 and win:  
  $F_3 = \{\text{Bankroll: 1020, Spins: [Red, Black, Red]}\}$

<br>

As can be seen from the above, each iteration of $F_{t}$ contains all information from the previous set plus whatever happened at time $t$. That is to say:

<br>

$$ F_0 ⊆ F_1 \subseteq F_2 ... ⊆ F_t $$
""", unsafe_allow_html=True
            )
