# Import necessary packages
from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns
import pandas as pd
import streamlit as st

def european_roulette_colour_betting(max_spins, starting_balance):
    spins = 0
    balance = starting_balance
    while balance > 0 and spins < max_spins:
        if np.random.rand() < 0.486:  # Probability for red/black in European roulette
            balance += 1
        else:
            balance -= 1
        spins += 1
    return spins

def simulate_bankruptcy(customers, max_spins, starting_balance):
    spins_to_bankruptcy = []
    for i in range(customers):
        customer_spins = european_roulette_colour_betting(max_spins, starting_balance)
        spins_to_bankruptcy.append(customer_spins)
        progress_bar.progress((i + 1) / customers)  # Update the progress bar
    return spins_to_bankruptcy, starting_balance

def plot_simulation(spins_to_bankruptcy, starting_balance):
    mean_spins = np.mean(spins_to_bankruptcy)
    var_spins = np.var(spins_to_bankruptcy, ddof=1)
    st_dev_spins = np.std(spins_to_bankruptcy, ddof=1)

    fig = plt.figure(figsize=(12, 6))
    sns.kdeplot(spins_to_bankruptcy, label=f'$\mathbb{{E}}_0[\\tau] =$ {mean_spins:.0f}, $\sigma_0^{2}[\\tau] = {var_spins:.0f}$, $\sigma_0[\\tau] = {st_dev_spins:.0f}$', clip=(100, None))
    plt.title(f"Simulation of spins to bankruptcy for starting balance = {starting_balance}")
    plt.xlabel('Number of Spins')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return fig

st.title("European Roulette Simulations")

st.write("""

_Note: The default plots and data on this page are saved runs from previous results, I have included the relevant code so you can
run your own simulations if you want but this can take some time to run._         

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



# sim_x100_stopping_dist = f"""
# spins_to_bankruptcy = []
# STARTING_BALANCE = 10
# CUSTOMERS = 1000
# MAX_SPINS = 5000

# for i in range(CUSTOMERS):
#   spin_list, customer_balance, customer_spins = european_roulette_colour_betting(MAX_SPINS,STARTING_BALANCE)
#   spins_to_bankruptcy.append(customer_spins)

# mean_spins = np.mean(spins_to_bankruptcy)
# var_spins = np.var(spins_to_bankruptcy, ddof = 1)
# st_dev_spins = np.std(spins_to_bankruptcy, ddof = 1)

# fig = plt.figure(figsize=(12, 6))
# sns.kdeplot(spins_to_bankruptcy, label=f'$\mathbb{{E}}_0[\\tau] =$ {mean_spins:.0f}, $\sigma_0^{2}[\\tau] = {var_spins:.0f}$, $\sigma_0[\\tau] = {st_dev_spins:.0f}$', clip=(100, None))
# plt.title(f"Simulation of spins to bankruptcy for starting balance = {STARTING_BALANCE}")
# plt.xlabel('Number of Spins')
# plt.ylabel('Density')
# plt.legend()
# plt.xlim(0,50000)


# st.pyplot(plt)
# """

import inspect



st.image("media/distribution_of_stopping_time_x0_100.png")

with st.expander("See Simulation Code"):
    st.code(inspect.getsource(european_roulette_colour_betting) + '\n' + inspect.getsource(simulate_bankruptcy) + '\n' + inspect.getsource(plot_simulation))


with st.expander("Run My Own Simulation"):
    graph_sims = st.slider("Select Number of Customers",
                       min_value=1000, max_value=10000, step=1000)
    
    # graph_trials = st.slider("Select Maximum Number of Spins",
    #                      min_value=1000, max_value=10000, step=1000)
    graph_starting_balance = st.slider("Select Starting Balance",
                                        min_value = 10, max_value = 100, step = 10
    )
    progress_bar = st.progress(0)  # Initialize the progress bar
    spins_to_bankruptcy, starting_balance_sim = simulate_bankruptcy(graph_sims, 50000, graph_starting_balance)
    sim_plot = plot_simulation(spins_to_bankruptcy, starting_balance_sim)
    st.pyplot(sim_plot)
    progress_bar = st.empty() 



st.markdown(r"""
So for this simulation we have the following summary statistics of the distribution of stopping times.

<br>
<center>

| Statistic | Value |
|:---:|:---:|
| Sample Mean, $\mu$ | 3727 |
| Sample Variance, $s^2$ | 5,244,957|
| Sample St Dev, $s$ | 2,290 |

</center>
<br>

But obviously this will change with every simulation, so to be sure this is a realistic estimate, I have simulated the same scenario but 10 times and take the average of the sample means and variances. 
Not sure if this is scientifically sound but it's just to validate any AI output.            
           """ , unsafe_allow_html=True)

# This takes a long time to run, save the output to drive and then load rather than simming every time


# df = pd.read_csv("data/data/sim_results.csv")
# df.rename(columns={df.columns[0]: "Summary Statistic" }, inplace = True)

# formatted_df = df.style.format('{:,.0f}')

# # Display the formatted DataFrame
# st.dataframe(formatted_df)

# # Display the type of the third column
# col_type = df.dtypes[2]

# st.write(f"The type of the third column is: {col_type}")