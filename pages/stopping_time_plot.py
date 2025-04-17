import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, isclose

# --- Simulation function ---
@st.cache_data
def run_simulation(bankroll, stake, net_payout, denom, churn_limit, n_players=100_000):
    """
    Run Monte Carlo for n_players, stopping at churn_limit or ruin.
    Returns array of stopping times and final bankrolls.
    """
    # initialise arrays
    banks = np.full(n_players, bankroll, dtype=float)
    stops = np.zeros(n_players, dtype=int)
    alive = np.ones(n_players, dtype=bool)
    
    p = 1.0 / denom
    q = 1.0 - p
    
    for spin in range(1, churn_limit + 1):
        if not alive.any():
            break
        # generate wins for active players
        r = np.random.random(size=n_players)
        win = (r < p) & alive
        
        # update banks: net gain on win, loss on others
        banks[win] += net_payout
        banks[alive & ~win] -= stake
        
        # mark those who just busted
        busted = alive & (banks < stake)
        stops[busted] = spin
        alive &= ~busted  # remove busted
        
        # for those still running, we'll record at the end if they hit churn_limit
        
    # any still alive at churn_limit
    stops[alive] = churn_limit
    return stops, banks

# --- Main dashboard ---
st.title("Casino Profitability under High‑Variance Games with Player Churn")

# Sidebar inputs
st.sidebar.header("Model inputs")
bankroll    = st.sidebar.slider("Player bankroll (£)",           1.0, 10_000.0, 100.0, step=1.0)
stake       = st.sidebar.slider("Stake per spin (£)",            0.10, 500.0,    1.0,   step=0.10)
net_payout  = st.sidebar.slider("Net payout on win (£)",         1,    1_000,   35,    step=1)
denom       = st.sidebar.slider("True win denominator (1 in n)", 2,    5_000,   37,    step=1)
churn_limit = st.sidebar.slider("Maximum spins before leave",    10,   5_000,   200,   step=1)

# Derived probabilities
p = 1.0 / denom
q = 1.0 - p

st.header("Analytic results (no churn limit)")

# 1) Expected spins to ruin: E[T] = bankroll / (q*stake - p*net_payout)
drift_per_spin = q * stake - p * net_payout
if drift_per_spin <= 0:
    E_T = np.inf
    sd_T = np.inf
else:
    E_T = bankroll / drift_per_spin
    # 2) Variance per spin: Var(X) = E[X^2] - (E[X])^2
    #    where X = +net_payout w.p. p, and -stake w.p. q
    EX2 = p * net_payout**2 + q * stake**2
    mu = p * net_payout - q * stake
    var_per_spin = EX2 - mu**2
    # 3) Var(T) ≈ var_per_spin * bankroll / drift_per_spin**3
    var_T = var_per_spin * bankroll / drift_per_spin**3
    sd_T = np.sqrt(var_T)

# 3) Probability of any win before bust: 1 - q**(bankroll/ stake)
#    (needs integer number of losses to ruin)
p_any_win = 1 - q**int(bankroll / stake)

# 4) Theoretical house edge %: (stake - p*(net_payout+stake))/stake
house_edge = (stake - p * (net_payout + stake)) / stake * 100

col1, col2, col3 = st.columns(3)
col1.metric("Exp. spins to ruin", f"{E_T:,.1f}")
col2.metric("SD of spins",         f"{sd_T:,.1f}")
col3.metric("P(any win before bust)", f"{p_any_win:.2%}")

st.write(f"**Theoretical house edge:** {house_edge:.2f}%")

st.header("Monte Carlo simulation (100 000 players)")

# run and cache simulation
st.write("Running simulation…")
st.spinner("Simulating spins...")
stop_times, final_banks = run_simulation(bankroll, stake, net_payout, denom, churn_limit)

# realised metrics
turnover = stop_times.sum() * stake
player_profit = final_banks.sum() - bankroll * 100_000
casino_profit = -player_profit
realised_margin = casino_profit / turnover * 100

col1, col2, col3 = st.columns(3)
col1.metric("Realised margin", f"{realised_margin:.2f}%")
col2.metric("Turnover (£)",      f"£{turnover:,.0f}")
col3.metric("Profit (£)",        f"£{casino_profit:,.0f}")

# density plot of stopping times
fig, ax = plt.subplots()
ax.hist(stop_times, bins=50, density=True, alpha=0.7)
ax.axvline(np.mean(stop_times),    linestyle='--', label='Mean stopping time')
ax.axvline(churn_limit, color='red', label='Churn limit')
ax.set_xlabel("Number of spins")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

# explanatory note
st.markdown(
    "Players lose on average over the lifetime of their bankroll. If the churn limit is **below** "
    "the mean time to ruin, many customers leave **before** the house edge can play out fully, "
    "so the *realised* margin falls below the *theoretical* edge per spin. Only when the limit "
    "exceeds the expected spins to ruin does the simulated margin converge on theory."
)