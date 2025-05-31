import numpy as np
import pandas as pd
import pytest
from utils.portfolio_env import PortfolioEnv

def create_mock_data(n_steps, n_assets, initial_price=100):
    """
    Helper function to create mock data for testing PortfolioEnv.
    """
    dates = pd.date_range(start='2023-01-01', periods=n_steps + 60) # Add buffer for window size

    raw_returns = np.random.rand(n_steps + 60, n_assets) * 0.02 - 0.01 # Small positive/negative returns

    # Ensure prices are positive
    prices_list = []
    current_prices = np.full(n_assets, initial_price)
    for i in range(n_steps + 60):
        prices_list.append(current_prices.copy())
        current_prices = current_prices * (1 + raw_returns[i, :])
        current_prices = np.maximum(current_prices, 1.0) # Ensure prices don't go to zero or negative

    prices = pd.DataFrame(prices_list, columns=[f'asset_{i}' for i in range(n_assets)], index=dates)
    returns = prices.pct_change().fillna(0)

    # Make vol_df larger to cover all steps including window_size
    vol = pd.DataFrame(np.random.rand(n_steps + 60, 3), columns=['vol1', 'vol2', 'vol3'], index=dates)

    # Slice to actual n_steps for prices_df, but keep returns_df and vol_df aligned with it
    # The environment uses .iloc[self.current_step] which can go up to len(returns_df) -1
    # So returns_df, prices_df, vol_df should be of the same length
    return returns, prices, vol

def test_portfolio_turnover():
    """
    Tests the portfolio turnover calculation in PortfolioEnv.
    """
    n_assets = 2
    window_size = 3 # Min window size for env due to vol_features slicing
    num_test_steps = 3 # Number of actions to take

    # We need enough data for window_size initialization and then num_test_steps
    # The environment's current_step starts at window_size.
    # If returns_df has length L, max current_step is L-1.
    # We need L-1 to be >= window_size + num_test_steps -1
    # So, L >= window_size + num_test_steps
    total_data_steps = window_size + num_test_steps

    returns_df, prices_df, vol_df = create_mock_data(n_steps=total_data_steps, n_assets=n_assets)

    env = PortfolioEnv(
        returns_df=returns_df,
        prices_df=prices_df,
        vol_df=vol_df,
        window_size=window_size,
        initial_balance=10000,
        transaction_cost=0 # Simplify by not having transaction costs
    )

    # Reset env and get initial weights
    obs, info = env.reset()
    # After reset, env.weights are [0,0,...] for assets. Let's call this w0.
    # env.weights_history is []. The step method will populate it.

    actions = [
        np.array([0.6, 0.4]), # Action for step 1 (target weights for rebalancing)
        np.array([0.7, 0.3]), # Action for step 2
        np.array([0.5, 0.5])  # Action for step 3
    ]

    # Store the sequence of actual weights applied for manual calculation
    # w0 is the state after reset.
    # In PortfolioEnv, weights are normalized by softmax.
    # The actions here are pre-softmax. The env applies softmax.

    # w0_actual = env.weights.copy() # These are all zeros for assets.
    # This is not added to env.weights_history by reset(), but by the first step()

    manual_weights_sequence = []
    # The first element in env.weights_history (added by the first step)
    # will be the initial weights (all zeros for assets).
    # Let's capture the state of weights as they evolve.

    # After reset, env.weights are the initial weights (e.g. [0,0] for assets).
    manual_weights_sequence.append(env.weights.copy())


    portfolio_values_history = [env.portfolio_value]
    for i, raw_action in enumerate(actions):
        # The env.step function applies softmax to raw_action
        obs, reward, terminated, truncated, info = env.step(raw_action)
        portfolio_values_history.append(info['portfolio_value'])
        # env.weights now holds the weights *after* rebalancing based on raw_action
        # This is w_i+1
        manual_weights_sequence.append(env.weights.copy())

    # manual_weights_sequence now contains [w0, w1, w2, w3]
    # w0 = initial weights from reset
    # w1 = weights after action 1 (softmax of raw_action[0])
    # w2 = weights after action 2 (softmax of raw_action[1])
    # w3 = weights after action 3 (softmax of raw_action[2])

    # Let's verify what env.weights_history contains
    # After 3 steps:
    # step 1: history.append(w0_reset), env.weights = w1
    # step 2: history.append(w1), env.weights = w2
    # step 3: history.append(w2), env.weights = w3
    # So env.weights_history = [w0_reset, w1, w2]
    # And env.weights = w3
    # The sequence for turnover should be [w0_reset, w1, w2, w3]
    # This is exactly env.weights_history + [env.weights.copy()]

    weights_for_calc = env.weights_history + [env.weights.copy()]

    assert len(weights_for_calc) == num_test_steps + 1
    assert np.array_equal(weights_for_calc[0], manual_weights_sequence[0]), "Mismatch in w0"
    assert np.array_equal(weights_for_calc[1], manual_weights_sequence[1]), "Mismatch in w1"
    assert np.array_equal(weights_for_calc[2], manual_weights_sequence[2]), "Mismatch in w2"
    assert np.array_equal(weights_for_calc[3], manual_weights_sequence[3]), "Mismatch in w3"


    # Manually calculate expected turnover
    # W = manual_weights_sequence = [w0, w1, w2, w3]
    # turnover = (sum(|w1-w0|) + sum(|w2-w1|) + sum(|w3-w2|)) / 3

    expected_turnover_sum = 0
    for i in range(1, len(manual_weights_sequence)):
        # Sum of absolute differences for each asset
        # weights are for assets only, cash is handled by w_c = 1 - sum(weights)
        # The turnover formula np.sum(np.abs(np.diff(np.array(weights_history), axis=0)))
        # considers all elements in the weight vectors. If cash is not part of these vectors,
        # its change is implicitly captured if asset weights change.
        # The weights in PortfolioEnv.weights are only for assets.
        diff = np.abs(manual_weights_sequence[i] - manual_weights_sequence[i-1])
        expected_turnover_sum += np.sum(diff)

    if num_test_steps > 0:
        expected_turnover = expected_turnover_sum / num_test_steps
    else:
        expected_turnover = np.nan # Or 0, consistent with env's calc_metrics

    metrics = env.calc_metrics(
        portfolio_values=np.array(portfolio_values_history),
        weights_history=weights_for_calc
    )
    calculated_turnover = metrics["Portfolio turnover"]

    if num_test_steps == 0:
        assert np.isnan(calculated_turnover), "Turnover should be NaN for no rebalancing periods"
    else:
        assert np.isclose(calculated_turnover, expected_turnover), \
            f"Calculated turnover {calculated_turnover} does not match expected {expected_turnover}"

def test_portfolio_turnover_no_steps():
    """
    Tests turnover calculation when there are no actual rebalancing steps, or only initial state.
    """
    n_assets = 2
    window_size = 3 # Min window size for env
    total_data_steps = window_size + 0 # No test steps beyond initialization

    returns_df, prices_df, vol_df = create_mock_data(n_steps=total_data_steps, n_assets=n_assets)
    env = PortfolioEnv(returns_df, prices_df, vol_df, window_size=window_size)

    obs, info = env.reset()

    # weights_history in calc_metrics will be env.weights_history + [env.weights.copy()]
    # After reset, env.weights_history is [], env.weights is [0,0]
    # If we call calc_metrics immediately:
    # final_weights_history = [env.weights.copy()] which is [[0,0]]
    # np.diff on this will be empty. Turnover should be NaN.
    metrics_initial = env.calc_metrics(
        portfolio_values=np.array([env.initial_balance, env.portfolio_value]), # dummy PV history
        weights_history=[env.weights.copy()] # Only initial weights
    )
    assert np.isnan(metrics_initial["Portfolio turnover"]), "Turnover should be NaN with only one set of weights"

    # If we take one step:
    # env.reset() -> w0 = [0,0], hist=[]
    # env.step(a1) -> hist=[w0], env.weights=w1
    # final_weights_history for calc_metrics = [w0, w1]
    # Turnover = sum(|w1-w0|) / 1
    obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5]))

    final_weights_one_step = env.weights_history + [env.weights.copy()] # Should be [w0, w1]
    assert len(final_weights_one_step) == 2

    expected_turnover_one_step = np.sum(np.abs(final_weights_one_step[1] - final_weights_one_step[0])) / 1.0

    metrics_one_step = env.calc_metrics(
        portfolio_values=np.array([env.initial_balance, env.portfolio_value, env.portfolio_value + 100]), # dummy
        weights_history=final_weights_one_step
    )
    assert np.isclose(metrics_one_step["Portfolio turnover"], expected_turnover_one_step)

# pytest.main() # For running with `python tests/test_portfolio_env.py`
# To run with pytest CLI, remove pytest.main() and just run `pytest` from root.
# For now, keeping it to allow direct execution.
