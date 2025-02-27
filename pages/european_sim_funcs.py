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
    """Check if the player's balance is significantly below expectations."""
    if len(balance_history) < 100:
        return False
    params = GAME_PARAMS[game]
    t = len(balance_history)
    center = params['centre'] * t
    std_dev = params['sigma'] * np.sqrt(t)
    lower_threshold = 100 + (center - 2 * std_dev)
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
    """
    Compute a "rolling margin" ratio.

    Given a list of casino_profit lists (each list corresponding to one simulation),
    this function creates a padded 2D array (using np.nan for missing values), then:
      - For column 0, uses the sum of valid entries.
      - For each subsequent column, adds the sum of differences (current - previous)
        for rows that have valid values.
      - Finally, the ratio is cumulative difference divided by the cumulative count
        of non-NaN values.
    """
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
    """
    Given a list of SimulationResult objects, extract:
      - overall rolling margin (from casino_profit, excluding the first value),
      - a list of spins,
      - closing balances,
      - total spins,
      - and grouping by stop_reason for individual rolling margins.
    Returns a dictionary of aggregated stats.
    """
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

        # Pick the max_spins based on the game
        if game == 'colours':
            max_spins = 20000
        else:  # 'numbers'
            max_spins = 100000

        sim_results = run_simulations(n=10000, max_spins=max_spins, starting_balance=100, game=game)
        simulation_output[game] = aggregate_results(sim_results)

    # Save the aggregated simulation output into a pickle file.
    with open('simulation_results_no_conditions.pickle', 'wb') as f:
        pickle.dump(simulation_output, f)

    print("\nSimulation results have been saved to 'simulation_results_no_conditions.pickle'.")