
import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np


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
**Stochastic Processes Theory**
""")

st.header("""
Introduction
""")

st.markdown(r"""
I want to examine the difference between quoted return to player (RTP) and achieved RTP when introducing budget constraints. 
            
First I need to review some of the mathematical theory and it appears both Stochastic Processes and Monte Carlo Simulations can both be used to model what I am interested in.

It has been a long time since I learnt any of this, so I'm going to start by going over some of the theory.
""")

st.header("""
          Simple Random Walk
""", divider=True)

st.subheader("""
          Proof of Expected Value
""")

st.markdown(r"""
For a random Independent and Identically Distributed variable

$$
Y_i \stackrel{iid}{\sim} \begin{cases}
1, & \text{with probability } 0.5 \\
-1, & \text{with probability } 0.5
\end{cases}
$$

for each $t$."""
            )

st.markdown(r"""

$$
            X_{t} = \displaystyle\sum_{i=1}^{t} Y_i
$$

and


$$
            X_{0} = 0
$$
""")

st.markdown(r"""
since the probability of both outcomes is 0.5, over infinite repeats we would expect the final balance for a player in a fair coin-toss game to be

$$
            E\left[X_t\right] = E\left[\displaystyle\sum_{i=1}^t Y_i\right] = \displaystyle\sum_{i=1}^t E\left[Y_i\right] = t \cdot 0 = 0
$$

since we have t repetitions of $E\left[Y_i\right]$ and
            
$$ 
            E\left[Y_i\right] = (-1) \cdot P(Y_i = -1) + (1) \cdot P(Y_i = 1) 
$$

$$
            = (-1) \cdot 0.5 + (1) \cdot 0.5 = 0
$$

            
""", unsafe_allow_html=True)

st.markdown(r"""
### Proof of Variance

#### 1. For Single Step

The equation for the variance of any random variable is given by the following equation:

 

$$
             \mathrm{Var}[{X}] = E[X^2] - (E[X])^2
$$

 

therefore the variance after a single step $Y_{i}$ is given by:

 

$$ 
            \mathrm{Var}(Y_{i}) = E[Y_{i}^2]-(E[Y_{i}])^2 
$$

 

$Y_{i}$ can only take two values, -1 or 1, or more formally:

 
$$ 
            Y_{i} \in \left\{ -1, 1 \right\} 
$$

 

therefore:

 

$$
            E[Y_{i}^2] = (-1)^2 \cdot P(Y_{i} = -1) + (1)^2 \cdot P(Y_{i} = 1) = 1 \cdot 0.5 + 1 \cdot 0.5 = 0.5 + 0.5 = 1
$$

 

and the mean value is given by:

 

$$ 
            E[Y_{i}] = (-1) \cdot 0.5 + 1 \cdot 0.5 = 0 
$$

 

so the square of the mean term in the variance equation is:

 

$$ 
            (E[Y_{i}])^2 = 0^2 = 0 
$$

 

and therefore substituting these into the variance formula yields:

 

$$ 
            \mathrm{Var}(Y_{i}) = E[Y_{i}^2]-(E[Y_{i}])^2 = 1 - 0^2 = 1 
$$

            
            """)

st.markdown(r"""
#### 2. For Any Step t

From the Expected Value proof we know that $ X_{t} = \displaystyle\sum_{i=1}^{t}Y_{i}$ and since the $Y_{i}$ are independent random variables:

 

$$ 
            \mathrm{Var}(X_{t}) = \mathrm{Var}\left(\displaystyle\sum_{i=1}^{t}Y_{i}\right) = \displaystyle\sum_{i=1}^{t}\mathrm{Var}(Y_{i}) 
$$

 

and since $\mathrm{Var}(Y_{i}) = 1$ for all $i$:

 

$$
             \mathrm{Var}(X_{t}) = \displaystyle\sum_{i=1}^{t} 1 = t 
$$

 

so the variance of the random walk $X_{t}$ at time $t$ is proportional to $t$. Note that this also means the standard deviation of $X_{t}$ is

 

$$
             \mathrm{Std}(X_{t}) = \sqrt{\mathrm{Var}(X_{t})} = \pm\sqrt{t}
$$
            """)

st.header("""
Python Visualisation
          """, divider=True)

st.markdown("""
As with any concept, I prefer to visualise with the use of graphs. I will start by defining a function that takes a single argument of number of trials
and returns a list of the state of our random walk for any time T = t
            """)

with st.expander("See Packages"):
    st.code('''
        import matplotlib.pyplot as plt
        import random
        import numpy as np
    ''')

with st.expander("See Random Walk Function"):
    st.code('''
def random_walk(n_trials):
  balance = [0,]
  t = [0,]
  running_balance = 0
  running_t = 0
  for i in range(n_trials):
    outcome = random.randint(0,1)
    if outcome == 0:
      running_balance -= 1
      balance.append(running_balance)
    else:
      running_balance += 1
      balance.append(running_balance)
    running_t += 1
    t.append(running_t)
  return t, balance
    ''')

random_walk_ss = st.slider(r"Select Number of Steps",
                           min_value=5, max_value=15, step=1)


list_1, list_2 = random_walk(random_walk_ss)
plt.plot(list_1, list_2)
fig, ax = plt.subplots()

# Plot the points
ax.plot(list_1, list_2)  # 'o-' adds points on each step

# Annotate arrows for direction
for i in range(1, len(list_1)):
    dx = list_1[i] - list_1[i-1]
    dy = list_2[i] - list_2[i-1]
    color = 'green' if dy > 0 else 'red'
    ax.annotate('', xy=(list_1[i], list_2[i]), xytext=(list_1[i-1], list_2[i-1]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1))

ax.spines['bottom'].set_visible(False)  # Hide bottom spine
ax.spines['left'].set_visible(False)   # Hide left spine
ax.spines['top'].set_visible(False)    # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

# Keep ticks but hide axes lines
ax.xaxis.set_ticks_position('bottom')  # Show ticks at the bottom
ax.yaxis.set_ticks_position('left')    # Show ticks at the left

ax.hlines(y=0, xmin=0, xmax=len(list_1), colors='black', linestyles='--')
ax.set_yticks(np.arange(int(min(list_2)), int(max(list_2)) + 1, 1))
ax.set_aspect('equal', 'box')

# Display the plot in Streamlit
st.pyplot(fig)


st.markdown("""
For a small values of t (steps) it is not easy to see that we expect the random walk to converge to zero. In some instances the deviation is as large as the number of steps.
However, if we increase the number of steps to 1000 we can see more clearly the behaviour of the random walk.
            """)

n1000_sim_plot = large_sim()

st.pyplot(n1000_sim_plot)

st.markdown(r"""
Here we are still only observing 5 simulations. We know that the expected value of t at any point is zero but what can we expect is a reasonable amount of noise?

The variance at any point on this graph is given by $t$ and therefore the standard deviation of the random walk is $\sqrt{t}$, meaning that we expect the majority of the random walks to be close in proximity to the bounds of $\sqrt{t}$. Furthermore, we expect these bounds to be crossed an infinite amount of times as $t \rightarrow \infty $

Let's demonstrate this visually. The graph below matches the random walk generated by the slider above but with some more labels to demonstrate the variance and the standard deviation. """)


def random_walk_annotated():
    root_t = [t**0.5 for t in list_1]
    minus_root_t = [-(t**0.5) for t in list_1]
    max_y = np.array([i for i in list_1])  # Convert max_y to a NumPy array
    min_y = np.array([-i for i in list_1])  # Convert min_y to a NumPy array

    fig, ax = plt.subplots()

    plt.plot(list_1, list_2)
    plt.plot(list_1, root_t, linestyle='--', alpha=0.5, color='black')
    plt.plot(list_1, minus_root_t, linestyle='--', alpha=0.5, color='black')
    plt.plot(list_1, max_y, alpha=0.75, color='black')
    plt.plot(list_1, min_y, alpha=0.75, color='black')
    plt.hlines(y=0, xmin=0, xmax=len(list_1), color='black')

    # plt.text(10, 9.2, 'y = t')
    # plt.text(10, -9.2, 'y = -t')
    # plt.text(10, 3.5, r'$y = \sqrt{t}$')
    # plt.text(10, -3.5, r'$y = -\sqrt{t}$')

    label_pos = len(list_1)
    plt.text(label_pos, label_pos*0.92, 'y = t')
    plt.text(label_pos, -label_pos*0.92, 'y = -t')
    plt.text(label_pos, label_pos**0.5, r'$y = \sqrt{t}$')
    plt.text(label_pos, -label_pos**0.5, r'$y = -\sqrt{t}$')

    plt.title(f'Random Walk Simulation for t = {len(list_1)-1}')
    plt.xlabel('Time (Steps)')
    plt.ylabel('Balance')
    ax = plt.gca()  # Get current axes
    ax.spines['bottom'].set_position(('data', 0))

    # Shading non-possible states of balance
    plt.fill_between(list_1, max_y, len(list_1)-1, where=(max_y <=
                     len(list_1)-1), color='lightcoral', alpha=0.3)
    plt.fill_between(list_1, min_y, -(len(list_1)-1),
                     where=(min_y >= -(len(list_1)-1)), color='lightcoral', alpha=0.3)

    plt.box(False)
    plt.show()
    return fig


annotated_sim = random_walk_annotated()

st.pyplot(annotated_sim)

st.markdown(r"""
As before, the assertion that we expected the walk to converge to zero and cross the st deviation lines an infinite amount of times will only become apparent when we plot multiple
simulations over a sufficiently large time frame.""")

graph_sims = st.slider("Select number of Simulations",
                       min_value=20, max_value=100, step=1)
graph_trials = st.slider("Select number of Trials",
                         min_value=1000, max_value=10000, step=1000)


def large_n_annotated(random_walk_function, num_simulations=25, n_trials=1000, sigma = 1):

    # Create an array of t values from 0 to max n_trials
    t_values = np.arange(0, n_trials+1)
    sqrt_t_values = np.sqrt(t_values)
    neg_sqrt_t_value = np.negative(sqrt_t_values)

    # Plot multiple simulations
    # Adjust figure size for better visualization
    fig = plt.figure(figsize=(12, 6))
    for _ in range(num_simulations):
        x, y = random_walk_function(n_trials)
        # Thinner lines and transparency for readability
        plt.plot(x, y, linewidth=0.4, alpha=0.4, color='gray')

    max_y = np.array([i for i in x])
    min_y = np.array([-i for i in x])

    # Plot the sqrt(t) and -sqrt(t) lines
    plt.plot(t_values, sqrt_t_values, color='black')
    plt.plot(t_values, neg_sqrt_t_value, color='black')
    plt.plot(t_values, max_y, color='black')
    plt.plot(t_values, min_y, color='black')

    plot_labels = len(t_values)
    plt.text(plot_labels*0.08, plot_labels*0.08, 'y = t')
    plt.text(plot_labels*0.08, -plot_labels*0.08, 'y = -t')
    plt.text(plot_labels, plot_labels ** 0.5, r'$y = \sqrt{t}$')
    plt.text(plot_labels, - plot_labels ** 0.5, r'$y = -\sqrt{t}$')

    plt.fill_between(x, max_y, len(x)-1, where=(max_y <=
                     len(x)-1), color='lightcoral', alpha=0.3)
    plt.fill_between(x, min_y, -(len(x)-1), where=(min_y >= -
                     (len(x)-1)), color='lightcoral', alpha=0.3)

    # Enhance the plot
    ax = plt.gca()  # Get current axes
    ax.spines['bottom'].set_position(('data', 0))  # Set x-axis to the middle
    plt.hlines(y=0, xmin=0, xmax=n_trials, color='black',
               linestyle='--')  # Add horizontal line at y=0
    plt.box(False)  # Remove plot box
    plt.ylim(-plot_labels*0.08, plot_labels*0.08)
    plt.title('Random Walk Simulations')
    plt.xlabel('Time (Steps)')
    plt.ylabel('Balance')
    plt.show()

    return fig


large_sim_plot = large_n_annotated(random_walk, graph_sims, graph_trials)

st.pyplot(large_sim_plot)

st.header("Random Walk with Drift", divider=True)

st.markdown(r"""
Thus far we have looked at a random walk with equal probability of moving in each direction.
            
For what I will be looking at, I am interested in a Random Walk with Drift.
A random walk is said to have drift if $X_n = X_0 + \sum_{i=1}^n X_i + n\mu$ where $\mu$ is the long term
 bias of the stochastic process.

A good real-life example of this type of random walk is colour betting on a European Roulette wheel.

In the first example we modelled a process in which the odds of winning or losing 1 unit was equal at a 50% probability. 
In roulette there are 37 pockets, 18 red and 18 black with one green pocket. The payout is the same as in our fair example,
however, the presence of the green pocket means the probability of being paid out for any choice of red or black is $\frac{18}{37}$, or in other words, slightly less than 50%.
            More formally:

#### Expected Value for Colour Choice
            
For a random Independent and Identically Distributed variable

 

$$
            Y_i \stackrel{iid}{\sim} \begin{cases}
1, & \text{with probability } \frac{18}{37} \\
-1, & \text{with probability } \frac{19}{37}
\end{cases}
$$

for each $t,$   


$$
            X_{t} = \displaystyle\sum_{i=1}^{t} Y_i
$$

and


$$
            X_{0} = 0
$$

 

The game is no longer fair and so we need to calculate the expected value of every individual spin

 

$$
             E\left[Y_i\right] = (-1) \cdot P(Y_i = -1) + (1) \cdot P(Y_i = 1)
$$

$$
            = (-1) \cdot \frac{19}{37} + (1) \cdot \frac{18}{37} = -\frac{1}{37} 
$$

 

And so the expected value of $X_{t}$ for any time t is given by:


 
$$
             E\left[X_t\right] = E\left[\displaystyle\sum_{i=1}^t Y_i\right] = \displaystyle\sum_{i=1}^t \left(E\left[Y_i\right]\right) = t \cdot -\frac{1}{37} = \frac{-t}{37}
$$

""")

st.markdown(r""""
#### Variance and Standard Deviation for Colour Choice

The equation for the variance of any random variable is given by the following equation:



$$ 
            \mathrm{Var}[{X}] = E[X^2] - (E[X])^2
$$



therefore the variance after a single step $Y_{i}$ is given by:



$$ 
            \mathrm{Var}(Y_{i}) = E[Y_{i}^2]-(E[Y_{i}])^2 
$$



again, $Y_{i}$ can only take two values, -1 or 35



$$
             Y_{i} \in \left\{ -1, 1 \right\} 
$$



therefore:



$$
            E[Y_{i}^2] = (-1)^2 \cdot P(Y_{i} = -1) + (1)^2 \cdot P(Y_{i} = 1) 
$$



$$
            = 1 \cdot \frac{18}{37} + 1 \cdot \frac{19}{37} 
$$



$$
             = 1 
$$



and the mean value is given by:



$$
             E[Y_{i}] = \frac{-1}{37} 
$$



so the square of the mean term in the variance equation is:



$$ 
            (E[Y_{i}])^2 = \left(\frac{-1}{37}\right)^2 = \frac{1}{1369} 
$$



and therefore substituting these into the variance formula yields:



$$ 
            \mathrm{Var}(Y_{i}) = 1 - \frac{1}{1369} = \frac{1368}{1369}  
$$



and



$$
             \mathrm{Std}(X_{i}) = \sqrt{\mathrm{Var}(X_{i})} = \pm\frac{6\sqrt{38}}{37} 
$$            
            """)

st.header("Python Visualisation", divider = True)

st.markdown(r"""
            Below we can see that for the same number of trials and simulations as the simple random walk (controlled by the same slider as earlier) we
             have a downward trend in the expected value of $Y_t$ and a process that now has a st.dev of $y = E[Y_{t}] - \sigma \sqrt{t}$
             """)

# Define the random walk function
def random_walk_with_drift_colours(n_trials):
    balance = [0]  # Start at zero
    running_balance = 0
    for _ in range(n_trials):
        outcome = random.randint(0, 36)
        running_balance += 1 if outcome <= 17 else -1
        balance.append(running_balance)
    return list(range(len(balance))), balance

# Number of simulations
def large_t_with_drift(num_simulations=25, n_trials=1000, pdf_view = False):

    all_drift = []

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size for better visualization
    for _ in range(num_simulations):
        x_drift, y_drift = random_walk_with_drift_colours(n_trials)
        ax.plot(x_drift, y_drift, linewidth=0.4, alpha=0.4, color='grey')
        all_drift.append(y_drift)
    
    # Compute statistics across simulations
    all_drift_array = np.array(all_drift)
    mean_drift = np.mean(all_drift_array, axis=0)
    sample_stdev_drift = np.std(all_drift_array, axis=0, ddof=1)
    sample_stdev_drift_ub = sample_stdev_drift + mean_drift
    sample_stdev_drift_lb = mean_drift - sample_stdev_drift

    # Define boundaries for y = t and y = -t
    max_y_drift = np.array(x_drift)   # y = t
    min_y_drift = -np.array(x_drift)    # y = -t

    # Standard deviation for a single bet
    sigma = 6 * np.sqrt(38) / 37

    # Define the drift lines
    t_values_drift = np.arange(0, n_trials + 1)
    expected_with_drift = -1/37 * t_values_drift
    pos_stdev_with_drift = expected_with_drift + sigma * np.sqrt(t_values_drift)
    neg_stdev_with_drift = expected_with_drift - sigma * np.sqrt(t_values_drift)

    # Plot the drift lines
    ax.plot(t_values_drift, pos_stdev_with_drift, color='black', label='Standard Deviation (Positive)')
    ax.plot(t_values_drift, neg_stdev_with_drift, color='black', label='Standard Deviation (Negative)')
    ax.plot(t_values_drift, expected_with_drift, color='black', linestyle='--', label='Expected Value')
    ax.hlines(y=0, xmin=0, xmax=len(t_values_drift), color='red', linestyle='--')
    ax.plot(t_values_drift, max_y_drift, color='black')
    ax.plot(t_values_drift, min_y_drift, color='black')

    # Instead of using len(x_drift)-1, use the same limits you'll enforce:
    desired_ymin = -0.06 * n_trials
    desired_ymax = 0.04 * n_trials
    ax.fill_between(x_drift, max_y_drift, desired_ymax,
                    where=(max_y_drift <= desired_ymax), color='lightcoral', alpha=0.3)
    ax.fill_between(x_drift, min_y_drift, desired_ymin,
                    where=(min_y_drift >= desired_ymin), color='lightcoral', alpha=0.3)

    # Place text labels (using coordinates within the desired range)
    ax.text(desired_ymax*1.3, desired_ymax * 0.8, r'$y = t$', fontsize=12)
    ax.text(desired_ymax*1.3, desired_ymin * 0.8, r'$y = -t$', fontsize=12)
    ax.text(n_trials * 0.9, expected_with_drift[-1] + sigma * np.sqrt(t_values_drift[-1]) + expected_with_drift[-1]*-0.2, 
            r'$y = E[Y_{t}] + \sigma \sqrt{t}$', fontsize=10)
    ax.text(n_trials * 0.9, expected_with_drift[-1] - sigma * np.sqrt(t_values_drift[-1]) - expected_with_drift[-1]*-0.2, 
            r'$y = E[Y_{t}] - \sigma \sqrt{t}$', fontsize=10)

    # Enhance the plot appearance
    ax.spines['bottom'].set_position(('data', 0))  # x-axis in the middle
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(desired_ymin, desired_ymax)
    
    # Disable autoscaling so these limits stick
    ax.autoscale(enable=False)
    
    ax.set_title('Random Walk Simulations')
    ax.set_xlabel('Time (Steps)')
    ax.set_ylabel('Balance')
    ax.set_frame_on(False)  # Similar to plt.box(False)
    
    return fig

large_n_with_drift_plot = large_t_with_drift(graph_sims, graph_trials)

st.pyplot(large_n_with_drift_plot)

st.markdown(r""" More generally, we can describe the distribution of possible values of $Y_t$ at any point t with the distribution
            
$$
N \sim \left(-\frac{t}{37}, \left(\frac{6\sqrt{38}}{37}\right)^2 t \right) 
$$

Using this, we can clean up the above view to remove the simulations and instead plot the corresponding PDFs at intervals of t
            """)

def pdf_view(n_trials=50000):
    # Parameters for multiple points
  t_values = [10000, 20000, 30000, 40000, 50000]  # Time steps to add normal distributions
  scaling_factor = 2000  # Adjust this to make the peaks more pronounced

      # Standard deviation for a single bet
  sigma = 6 * np.sqrt(38) / 37

  # Define the drift lines
  t_values_drift = np.arange(0, n_trials + 1)
  expected_with_drift = -1/37 * t_values_drift
  pos_stdev_with_drift = expected_with_drift + sigma * np.sqrt(t_values_drift)
  neg_stdev_with_drift = expected_with_drift - sigma * np.sqrt(t_values_drift)

  # Plot the main lines
  fig = plt.figure(figsize=(12, 6))
  plt.plot(t_values_drift, pos_stdev_with_drift, color='black', label='Standard Deviation (Positive)')
  plt.plot(t_values_drift, neg_stdev_with_drift, color='black', label='Standard Deviation (Negative)')
  plt.plot(t_values_drift, expected_with_drift, color='black', linestyle='--', label='Expected Value')


  plt.text(5000, 500, r'$y = t$')
  plt.text(5000,-2000, r'$y = -t$')
  plt.text(42000, -500, r'$y = E[Y_{t}] + \sigma \sqrt{t}$')
  plt.text(42000, -1800, r'$y = E[Y_{t}] - \sigma \sqrt{t}$')


  # Add normal distributions at specified time steps
  for t in t_values:
      # Compute center (mean) and standard deviation
      center = -1 / 37 * t
      std_dev = sigma * np.sqrt(t)

      # Generate normal distribution
      y_values = np.linspace(center - 3 * std_dev, center + 3 * std_dev, 1000)
      normal_dist = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((y_values - center) ** 2) / (2 * std_dev**2))

      # Normalize and scale for visualization
      normal_dist_scaled = normal_dist / max(normal_dist) * scaling_factor

      # Plot the normal distribution
      plt.plot(t + normal_dist_scaled-1500, y_values, color='red', label=f'Normal Distributions' if t == 10000 else None)

  x_drift = np.arange(n_trials + 1)
  max_y_drift = np.array(x_drift) 
  min_y_drift = -np.array(x_drift)

  plt.fill_between(x_drift, max_y_drift, len(x_drift)-1, where=(max_y_drift <= len(x_drift)-1), color='lightcoral', alpha=0.3)
  plt.fill_between(x_drift, min_y_drift, -(len(x_drift)-1), where=(min_y_drift >= -(len(x_drift)-1)), color='lightcoral', alpha=0.3)

  plt.plot(t_values_drift, max_y_drift, color = 'black')
  plt.plot(t_values_drift, min_y_drift, color = 'black')


  # Add generic formula to the title
  generic_formula = r'$N \sim \left(-\frac{t}{37}, \left(\frac{6\sqrt{38}}{37}\right)^2 t \right)$'
  plt.title(f'Random Walk Simulation for European Roulette Follows Distribution: {generic_formula}', fontsize=14)

  plt.xlabel('Time (Steps)')
  plt.ylabel('Balance')
  plt.ylim(-2500, 1000)
  plt.legend()
  return fig

pdf_plot = pdf_view()

st.pyplot(pdf_plot)
       