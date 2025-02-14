
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
In the introduction to stochastic processes we looked at modelling the European Roulette wheel as a stochastic process and analysed some of the properties of the expected distribution of final player balances.

A base assumption was that both the casino and the customer would continue to transact infinitely which is not true in practice. If we assume that
the customer is constricted by some budget. If the stochastic process is stopped by the balance hitting an upper or lower bound, this is known as a _stopping time._
""")

st.header("""
          Stopping Times
""", divider=True)

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

st.markdown(r"""
## Practical Examples

We can calculate the expected amount of time until the player has spent all of a given budget. This is
 known as the _gambler's ruin problem._

### 1.1 Bankruptcy Colours Betting

The obvious example we can use this theory for is the case in which the gambler loses all of their initial bankroll.

We will start by setting the customer's balance at £100 or $X_0 = 100$, what we're looking to find is where $\mathbb{E}[X_{t}] = 0$. We know that the absolute minimum time this could happen for £1 stakes would be 100 spins but winnings could greatly increase the time in which we expected to achieve a balane of zero. What we're looking for is the expected value of the stopping time,  $\mathbb{E}[\tau]$

As we've seen before this can be written as a random walk like so:

<br>

$$
X_{t+1} =  \begin{cases}
X_{t} + b, & \text{with probability } p \\
X_{t} - b, & \text{with probability } 1-p
\end{cases}
$$

where:
""", unsafe_allow_html=True
            )

st.write(r"""
- $p$ is the probability of success, in this case correctly identifying a number
- $b$ is the bet size (and also the return size in the case of betting on colours)
- $X_t$ is the gambler's balance at time $t$
""")

st.markdown(r"""

**Expected Change Per Step (Drift)**

The drift $ \Delta $ is the expected change in the gambler's bankroll after one step:

<br>

$$ \Delta = \mathbb{E}[X_{t+1} - X_{t}] = b \cdot p - b \cdot (1 - p). = bp - b + bp $$

<br>

$$
= 2bp - b = b(2p-1)
$$

<br>

The formula for the expected number of steps before bankruptcy is given by

<br>

$$
\mathbb{E}[\tau] = \frac{X_0}{|b\cdot(2p-1)|}
$$

<br>

which for £1 stakes, a starting balance of £100 and a win probability of $\frac{18}{37}$ gives us

<br>

$$
\mathbb{E}[\tau] = \frac{X_0}{b\cdot(1-2p)} = \frac{100}{1\cdot\left(1-2\cdot\left(\frac{18}{37}\right)\right)} = \frac{100}{0.027} \approx 3703
$$
""", unsafe_allow_html=True
            )


st.markdown(r"""
### 1.2 Betting on Numbers

In the previous example, the outcomes of the random walk were the previous balance $\pm 1$ because betting on colours will return your stake plus an additional unit. In the example of betting on numbers, you receive your stake plus 35 additional units, and therefore, the process can be modeled as:

$$ 
X_{t+1} =  \begin{cases}
X_{t} + (R \cdot b), & \text{with probability } p \\
X_{t} - b, & \text{with probability } 1-p
\end{cases}
$$

where:
""", unsafe_allow_html=True)

st.write(
r"""
- $p$ is the probability of success, in this case, correctly identifying a number.
- $b$ is the bet size.
- $R$ is the return multiplier.
- $X_t$ is the gambler's balance at time $t$.
"""
)

st.markdown(r"""
**Expected Change Per Step (Drift)**

The drift $ \Delta $ is the expected change in the gambler's bankroll after one step:

$$ \Delta = \mathbb{E}[X_{t+1} - X_{t}] = b \cdot p \cdot R  - b \cdot (1 - p). = bpR - b + bp $$

$$
= 2bp - b = b(2p-1)
$$

The formula for the expected number of steps before bankruptcy is given by

$$
\mathbb{E}[\tau] = \frac{X_0}{|b\cdot(R \cdot p-1)|}
$$

which for £1 stakes, a starting balance of £100, a win probability of $\frac{1}{37}$, and a payout multiplier of 36 gives us

$$
\mathbb{E}[\tau] = \frac{X_0}{|b\cdot(R \cdot p-1)|}  = \frac{100}{|1\cdot\left(36 \cdot \left(\frac{1}{37} -1\right)\right)|} = \frac{100}{0.027} \approx 3703
$$

**Hold on** that's the same result that we had for betting on colours. 
            
This is perhaps to be expected since the individual expected result of each spin is the same, and so the expected time to bankruptcy 
should also be the same. We know in practice customers are likely to be sensitive to big swings in balance, so the *variance* of
stopping times might be more important to analyze.

The whole distribution would be useful, however, I'm not finding good information on how
to derive this so I will pause for the time being and take a simulation based approach.
""", unsafe_allow_html=True)


st.markdown(r"""
## Deepseek

The following answer for variance came from DeepSeek and looks to align with the result I got from simulations.

The variance of stooping time in the gambler's ruin problem can be computed using the formula:

<br>

$$
             \mathrm{Var}(\tau) = \frac{X_0 \cdot (1 - (q - p)^2)}{(q - p)^3} 
$$

<br>

I would like to find a proof for this before I base inference using it. For the time being let's just substitute in the values we're interested in. We know from the expected formula that $ q - p = \frac{19}{37} - \frac{18}{37} = \frac{1}{37} = 0.027$ therefore we have:

<br>

$$
             \mathrm{Var}(\tau) = \frac{100 \cdot (1 - (0.027)^2)}{(0.027)^3} = 5,075,000 \text{ spins}
$$

<br>

and for standard deviation

<br>

$$ 
\mathrm{St.Dev}(\tau) = \sqrt{5,075,000} = 2252.8 \text{ spins}
$$

<br>

Which aligns nicely with our simulated outcomes. But then what if we use the same formula for number betting where $q-p = \frac{36}{37}-\frac{1}{37} = \frac{35}{37}$

<br>

$$
 \mathrm{Var}(\tau) = \frac{100 \cdot (1 - (0.945)^2)}{(0.945)^3} = 12.68 \text{ spins} 
$$
""",unsafe_allow_html=True)


