# Import necessary packages
from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns
import pandas as pd
import streamlit as st

# Define the colours betting game with a list to track balance and number of spins (t)
def european_roulette_colour_betting(max_number_of_spins, starting_balance):
    balance = []
    t = [0]
    running_balance = starting_balance
    running_t = 0

    for _ in range(max_number_of_spins):
        if running_balance <= 0:  # Stop if bankrupt
            break
        outcome = random.randint(0, 36)

        # Assume 18/37 outcomes are wins (e.g., red or black)
        if outcome in range(0, 18):  # Winning outcomes
            running_balance += 1  # Win pays 1 unit
        else:
            running_balance -= 1  # Lose 1 unit

        # Track balance and time
        balance.append(running_balance)
        running_t += 1
        t.append(running_t)

    spins_to_bankruptcy = running_t if running_balance <= 0 else max_number_of_spins  # None if not bankrupt
    return t, balance, spins_to_bankruptcy
st.title("European Roulette Simulations")

st.write("""

## **Introduction**
After the first attempt to simulate data for calculating stopping times I felt it would be better to use a statistical approach so that I could generate a distribution for the expected stopping time (bankruptcy or a top threshold). However, I have not been able to find a good source that shows how to do this.

One way is to simulate the amount of time taken to hit the stopping condition and fitting a distribution to that data.

### **Warning**
While the expectation for any supermartingale is that eventually the gambler's balance will go to zero, this could take a very large number of spins. I would want to restrict the number of spins a gambler can have as I did in the first simulation notebook. I don't know how this will affect the distribution (other than to create a second peak at the spin limit where variability is high).
""")

st.markdown(r"""
  

## **Colours Betting**

Take the basic example of betting on colours with a starting balance of $X_0$ and £1 spins, how long will it take for the customer to reach zero on average?

We know from the Stochastic Processes work that for any value of $X_0$

<br>

$$
\mathbb{E}[\tau] = \frac{X_0}{b\cdot(1-2p)} = \frac{X_0}{1\cdot\left(1-2\cdot\left(\frac{18}{37}\right)\right)} = \frac{X_0}{0.027}
$$



<br>

But the solution for $\mathrm{Var}[\tau]$ is less clear.    

 We know that for a starting balance of £100 from the formula the expected number of spins is $\frac{100}{0.027} = 3703$ but trying to find an accurate answer 
for what the variance is wasn't as simple so instead I'll take the sample variance and st.dev from a simulation for 10,000 customers. 
      """, unsafe_allow_html=True)

spins_to_bankruptcy = []
STARTING_BALANCE = 100
CUSTOMERS = 10000
MAX_SPINS = 50000

for i in range(CUSTOMERS):
  spin_list, customer_balance, customer_spins = european_roulette_colour_betting(MAX_SPINS,STARTING_BALANCE)
  spins_to_bankruptcy.append(customer_spins)

mean_spins = np.mean(spins_to_bankruptcy)
var_spins = np.var(spins_to_bankruptcy, ddof = 1)
st_dev_spins = np.std(spins_to_bankruptcy, ddof = 1)

fig = plt.figure(figsize=(12, 6))
sns.kdeplot(spins_to_bankruptcy, label=f'$\mathbb{{E}}_0[\\tau] =$ {mean_spins:.0f}, $\sigma_0^{2}[\\tau] = {var_spins:.0f}$, $\sigma_0[\\tau] = {st_dev_spins:.0f}$', clip=(100, None))
plt.title(f"Simulation of spins to bankruptcy for starting balance = {STARTING_BALANCE}")
plt.xlabel('Number of Spins')
plt.ylabel('Density')
plt.legend()
plt.xlim(0,50000)

st.pyplot(plt)

st.markdown(r"""
So for this simulation we have the following summary statistics of the distribution of stopping times.

<br>
<center>

| Statistic | Value |
|:---:|:---:|
| Sample Mean, $\mu$ | 3707 |
| Sample Variance, $s^2$ | 4,937,662|
| Sample St Dev, $s$ | 2,222 |

</center>
<br>

But obviously this will change with every simulation, so to be sure this is a realistic estimate, I will simulate the same scenario but 10 times and take the average of the sample means and variances. Not sure if this is scientifically sound but it's just to validate any AI output.            
           """ , unsafe_allow_html=True)

# This takes a long time to run, save the output to drive and then load rather than simming every time


df = pd.read_csv("data/data/sim_results.csv")
pd.options.display.float_format = '{:,.0f}'.format 
df