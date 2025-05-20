import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from typing import Dict, Tuple, Optional, Union, List, Any


class PortfolioEnv(gym.Env):
    """
    A Gymnasium-compatible environment for portfolio optimization using Deep Reinforcement Learning (DRL).

    This environment simulates a trading scenario where an agent needs to decide how to
    allocate money across different assets to maximize returns while managing risk.

    The reward here is the Differential Sharpe Ratio (DSR),
    which is a measure of the risk-adjusted return of the portfolio

    State space (flattened 1D vector) includes:
    - Flattened historical log returns matrix (window_size, n_risky_assets)
    - Flattened tiled global volatility features (window_size, 3_vol_features)
      (Vol features assumed: S&P500_vol20, S&P500_vol20/vol60_ratio, VIX_index)
    - Current portfolio weights for risky assets (n_risky_assets,)
    - Current cash weight (1,)

    Action space:
    - Continuous portfolio weights for risky assets (sum to 1 after softmax)
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,  # Log returns for risky assets
        prices_df: pd.DataFrame,  # Prices for risky assets
        vol_df: pd.DataFrame,  # Global volatility features (standardized)
        window_size: int = 60,
        transaction_cost: float = 0,
        initial_balance: float = 100_000,
        reward_scaling: float = 1.0,
        eta: float = 1 / 252,  # smooth param for DSR
    ):
        """
        Initialize the portfolio environment with historical data and parameters.
        """
        super().__init__()

        # Store data
        self.returns_df = returns_df
        self.prices_df = prices_df
        self.vol_df = vol_df
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling

        # Exponential moving average for DSR
        self.eta = eta
        self.A_t = 0.0
        self.B_t = 0.0
        self.prev_A_t = 0.0
        self.prev_B_t = 0.0

        # Get number of risky assets
        self.n_risky_assets = len(returns_df.columns)
        if self.n_risky_assets != len(prices_df.columns):
            raise ValueError(
                "returns_df and prices_df must have the same number of asset columns."
            )
        if self.vol_df.shape[1] != 3:
            # Assuming 3 global volatility features as per paper's description
            # S&P500_vol20, S&P500_vol20/vol60_ratio, VIX_index
            print(
                f"Warning: vol_df has {self.vol_df.shape[1]} columns. Expected 3 global volatility features."
            )

        # Define action space (portfolio weights for risky assets)
        # Agent outputs scores; environment applies softmax
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_risky_assets,), dtype=np.float32
        )

        # Define observation space (flattened)
        # log_returns_flat: window_size * n_risky_assets
        # vol_features_flat: window_size * 3
        # current_risky_weights: n_risky_assets
        # current_cash_weight: 1
        observation_dim = (
            (self.window_size * self.n_risky_assets)
            + (self.window_size * 3)
            + self.n_risky_assets
            + 1
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

        # Portfolio state
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        # self.weights stores actual realized weights for risky assets after discrete rebalancing
        self.weights = np.zeros(self.n_risky_assets)
        self.cash_weight = 1.0 - np.sum(self.weights)
        self.actual_shares_risky = np.zeros(self.n_risky_assets)
        self.cash = self.initial_balance

        # RESET
        self.reset()

    def calc_metrics(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any]:
        if isinstance(portfolio_values, (list, pd.Series)):
            portfolio_values = np.array(portfolio_values)
        if len(portfolio_values) < 2:
            return {
                metric: 0.0
                for metric in [
                    "Annual return",
                    "Cumulative returns",
                    "Annual volatility",
                    "Sharpe ratio",
                    "Calmar ratio",
                    "Stability",
                    "Max drawdown",
                    "Omega ratio",
                    "Sortino ratio",
                    "Skew",
                    "Kurtosis",
                    "Tail ratio",
                    "Daily value at risk",
                    "Portfolio turnover",
                ]
            }  # Return default if not enough data

        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if (
            len(daily_returns) == 0
        ):  # handles portfolio_values with only 1 or 2 identical elements
            daily_returns = np.array([0.0])

        annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (
            252 / len(portfolio_values)
        ) - 1
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_volatility = np.std(daily_returns) * np.sqrt(252)

        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio_denominator = np.std(daily_returns)
        sharpe_ratio = (
            np.sqrt(252) * np.mean(excess_returns) / sharpe_ratio_denominator
            if sharpe_ratio_denominator != 0
            else 0
        )

        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        negative_returns_std_dev = (
            np.std(daily_returns[daily_returns < 0])
            if len(daily_returns[daily_returns < 0]) > 0
            else 0
        )
        sortino_ratio = (
            np.sqrt(252) * np.mean(excess_returns) / negative_returns_std_dev
            if negative_returns_std_dev != 0
            else 0
        )

        threshold = risk_free_rate / 252
        sum_positive_excess = np.sum(
            daily_returns[daily_returns > threshold] - threshold
        )
        sum_negative_excess = abs(
            np.sum(daily_returns[daily_returns <= threshold] - threshold)
        )
        omega_ratio = (
            sum_positive_excess / sum_negative_excess
            if sum_negative_excess != 0
            else float("inf")
        )

        skew = pd.Series(daily_returns).skew() if len(daily_returns) > 0 else 0
        kurtosis = pd.Series(daily_returns).kurtosis() if len(daily_returns) > 0 else 0

        percentile_5 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        percentile_95 = (
            np.percentile(daily_returns, 95) if len(daily_returns) > 0 else 0
        )
        tail_ratio = percentile_95 / abs(percentile_5) if percentile_5 != 0 else 0

        var = percentile_5  # VaR 95% is the 5th percentile

        # annual_volatility can be nan if std(daily_returns) is 0
        stability = 1 / (1 + annual_volatility) if annual_volatility is not None else 0

        # TODO Portfolio turnover ?

        return {
            "Annual return": annual_return,
            "Cumulative returns": cumulative_return,
            "Annual volatility": annual_volatility,
            "Sharpe ratio": sharpe_ratio,
            "Calmar ratio": calmar_ratio,
            "Stability": stability,
            "Max drawdown": max_drawdown,
            "Omega ratio": omega_ratio,
            "Sortino ratio": sortino_ratio,
            "Skew": skew,
            "Kurtosis": kurtosis,
            "Tail ratio": tail_ratio,
            "Daily value at risk": var,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Start after enough data for first observation window
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.weights = np.zeros(self.n_risky_assets)
        self.cash_weight = 1
        self.cash = 100_000

        # DSR state reset
        self.A_t = 0.0
        self.B_t = 0.0
        self.prev_A_t = 0.0
        self.prev_B_t = 0.0

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # --- 1. Apply Action: Determine Target Allocation ---
        if np.isnan(action).any():
            raise ValueError("NaN values in action")

        exp_action = np.exp(action - np.max(action))  # Softmax for numerical stability
        target_continuous_weights_risky = exp_action / np.sum(exp_action)

        # --- 2. Rebalance Portfolio (Discrete Shares) ---
        portfolio_value_at_start_of_step = self.portfolio_value
        current_prices_risky = self.prices_df.iloc[self.current_step].values

        if np.any(current_prices_risky <= 0):  # Check for invalid prices
            print(
                f"Warning: Invalid prices at step {self.current_step}: {current_prices_risky}"
            )
            # Default to holding previous portfolio if prices are invalid
            # This might lead to issues, proper handling depends on data quality
            target_continuous_weights_risky = self.weights

        # Calculate value to allocate to each risky asset
        target_risky_asset_values = (
            target_continuous_weights_risky * portfolio_value_at_start_of_step
        )

        # Calculate target shares (can be fractional)
        # Add a small epsilon to current_prices_risky to avoid division by zero if a price is exactly 0
        target_shares_risky = target_risky_asset_values / (current_prices_risky + 1e-9)

        # Actual shares (rounded down)
        new_actual_shares_risky = np.floor(target_shares_risky)

        # Value actually invested in risky assets & remaining cash
        value_invested_in_risky = np.sum(new_actual_shares_risky * current_prices_risky)
        new_cash = portfolio_value_at_start_of_step - value_invested_in_risky

        # Update portfolio holdings
        self.actual_shares_risky = new_actual_shares_risky
        self.cash = new_cash

        # Update actual weights based on discrete rebalancing
        if portfolio_value_at_start_of_step > 0:
            self.weights = (
                self.actual_shares_risky * current_prices_risky
            ) / portfolio_value_at_start_of_step
            self.cash_weight = self.cash / portfolio_value_at_start_of_step
        else:  # Should not happen if initial_balance > 0 and no catastrophic losses
            self.weights = np.zeros(self.n_risky_assets)
            self.cash_weight = 1.0

        # --- 3. Calculate Actual Portfolio Return for this step ---
        # Use prices at the END of the current step (which are start of next step for valuation)
        # self.current_step points to the data row for the *current day's decision and rebalancing*.
        # The return is realized based on *next day's opening prices* or *current day's closing prices*.
        # Assuming prices_df.iloc[self.current_step] are closing prices for day `t`.
        # Then returns_df.iloc[self.current_step] are returns from t-1 close to t close.
        # If rebalancing happens at `t` close, then value evolves until `t+1` close.
        # The log returns in self.returns_df.iloc[self.current_step] are for period (t-1 to t).
        # The DSR calculation uses the portfolio return for *this* step.
        # Let's use prices at t+1 to calculate portfolio value at end of step.
        if self.current_step + 1 >= len(self.prices_df):  # Cannot get next day's price
            portfolio_value_at_end_of_step = (
                portfolio_value_at_start_of_step  # No change if last day
            )
            actual_portfolio_return = 0.0
        else:
            prices_at_end_of_step_risky = self.prices_df.iloc[
                self.current_step + 1
            ].values
            value_of_risky_at_end_of_step = np.sum(
                self.actual_shares_risky * prices_at_end_of_step_risky
            )
            portfolio_value_at_end_of_step = value_of_risky_at_end_of_step + self.cash

            if portfolio_value_at_start_of_step > 1e-9:  # Avoid division by zero
                actual_portfolio_return = (
                    portfolio_value_at_end_of_step / portfolio_value_at_start_of_step
                ) - 1.0
            else:
                actual_portfolio_return = 0.0

        self.portfolio_value = portfolio_value_at_end_of_step  # Update portfolio value

        # --- 4. Calculate Reward (DSR) ---
        # DSR uses the actual portfolio return realized in this step
        if np.isnan(actual_portfolio_return):
            # This can happen if portfolio_value_at_start_of_step was ~0 or prices were problematic
            print(
                f"Warning: NaN in actual_portfolio_return at step {self.current_step}. Using 0.0."
            )
            actual_portfolio_return = 0.0
            # Consider a penalty or alternative handling if this occurs frequently.

        reward = self._calculate_reward(actual_portfolio_return)

        # Update DSR moving averages for *next* step's DSR calculation
        self.prev_A_t = self.A_t
        self.prev_B_t = self.B_t
        self.A_t = self.prev_A_t + self.eta * (actual_portfolio_return - self.prev_A_t)
        self.B_t = self.prev_B_t + self.eta * (
            actual_portfolio_return**2 - self.prev_B_t
        )

        if np.isnan(self.A_t) or np.isnan(self.B_t):
            raise ValueError(f"NaN in DSR moving averages at step {self.current_step}")

        # --- 5. Prepare for Next Step ---
        self.current_step += 1
        terminated = (
            self.current_step >= len(self.returns_df) - 1
        )  # -1 because we need t+1 prices for return calc
        truncated = False  # Or implement based on some other condition if needed

        observation = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "actual_portfolio_return": actual_portfolio_return,
            "actual_risky_weights": self.weights.copy(),
            "cash_weight": self.cash_weight,
            "target_continuous_weights_risky": target_continuous_weights_risky.copy(),
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        if self.current_step < self.window_size:
            # Should not happen if reset correctly sets current_step = window_size
            # Or handle by padding if necessary for very start of an episode
            start_idx = 0
        else:
            start_idx = self.current_step - self.window_size

        end_idx = self.current_step

        # Historical log returns for risky assets
        returns_window_risky = self.returns_df.iloc[
            start_idx:end_idx
        ].values  # (window_size, n_risky_assets)
        if (
            returns_window_risky.shape[0] < self.window_size
        ):  # Pad if at beginning of data
            padding = np.zeros(
                (self.window_size - returns_window_risky.shape[0], self.n_risky_assets)
            )
            returns_window_risky = np.vstack((padding, returns_window_risky))

        # Global volatility features (for current day, tiled back across window)
        # Vol features are from vol_df.iloc[self.current_step] (i.e. for day t)
        # To match the paper's "log returns over T ... vol indicators at time t",
        # these vol indicators are current. Tiling them is one way to fit the window structure.
        current_vol_features = self.vol_df.iloc[
            end_idx - 1
        ].values  # Use vol data for day t-1 (end_idx-1) if current_step is t
        # or end_idx if current_step is t and we use features of day t
        # Let's assume vol_df.iloc[k] are features for day k
        # If current_step is t, we need features for decision at t
        # The state is St. So vol_df.iloc[self.current_step]
        current_vol_features = self.vol_df.iloc[self.current_step].values  # (3,)
        vol_features_window = np.tile(
            current_vol_features, (self.window_size, 1)
        )  # (window_size, 3)

        # Flattened windowed data
        flat_returns_window = returns_window_risky.flatten()
        flat_vol_features_window = vol_features_window.flatten()

        # Current portfolio weights (realized weights for risky assets + cash weight)
        # self.weights and self.cash_weight are from the rebalancing at the start of current_step
        current_risky_weights_arr = self.weights
        current_cash_weight_arr = np.array([self.cash_weight])

        # Concatenate all parts into a 1D observation vector
        observation = np.concatenate(
            [
                flat_returns_window,
                flat_vol_features_window,
                current_risky_weights_arr,
                current_cash_weight_arr,
            ]
        ).astype(np.float32)

        if np.isnan(observation).any():
            # Try to find where NaN originates
            if np.isnan(returns_window_risky).any():
                print(f"NaN in returns_window_risky at step {self.current_step}")
            if np.isnan(vol_features_window).any():
                print(
                    f"NaN in vol_features_window (from current_vol_features) at step {self.current_step}"
                )
            if np.isnan(current_risky_weights_arr).any():
                print(f"NaN in current_risky_weights_arr at step {self.current_step}")
            if np.isnan(current_cash_weight_arr).any():
                print(f"NaN in current_cash_weight_arr at step {self.current_step}")
            raise ValueError(f"NaN values in observation at step {self.current_step}")

        return observation

    def _calculate_reward(self, actual_portfolio_return: float) -> float:
        # If first step after reset (where current_step was set to window_size),
        # prev_A_t and prev_B_t are 0.
        if (
            self.current_step == self.window_size
        ):  # First step where DSR can be meaningfully calculated
            # No DSR can be computed meaningfully as A_tm1 and B_tm1 are likely 0 or uninitialized
            # Or, initialize A_t, B_t with a burn-in period if desired.
            # For now, return 0 for the very first calculation.
            return 0.0

        A_tm1 = self.prev_A_t  # EMA of returns up to t-1
        B_tm1 = self.prev_B_t  # EMA of squared returns up to t-1

        delta_A = actual_portfolio_return - A_tm1
        delta_B = actual_portfolio_return**2 - B_tm1

        if np.isnan(delta_A) or np.isnan(delta_B):
            # This should be caught by NaN check on actual_portfolio_return earlier
            raise ValueError(f"NaN in DSR deltas at step {self.current_step}")

        # Differential Sharpe Ratio (Moody et al. 1998)
        numerator_dsr = B_tm1 * delta_A - 0.5 * A_tm1 * delta_B
        denominator_dsr_squared = B_tm1 - A_tm1**2

        # Denominator for DSR is (B_t-1 - A_t-1^2)^(3/2)
        # Add numerical stability: ensure (B_tm1 - A_tm1**2) is non-negative and denominator is not too small
        if denominator_dsr_squared < 1e-9:  # If variance is close to zero or negative
            # If variance is zero (constant returns), DSR is undefined or could be treated as 0.
            # A negative value here indicates B_tm1 < A_tm1^2, which is problematic.
            # This might happen early on if returns are very small.
            dsr = 0.0
        else:
            denominator_dsr_pow_3_2 = denominator_dsr_squared**1.5
            if denominator_dsr_pow_3_2 < 1e-9:  # Denominator of DSR itself is too small
                dsr = 0.0  # Or a large penalty/value depending on numerator sign
            else:
                dsr = numerator_dsr / denominator_dsr_pow_3_2

        if np.isnan(dsr):
            # This can happen if A_tm1, B_tm1 lead to issues.
            # E.g. if B_tm1 - A_tm1**2 is negative.
            print(
                f"Warning: NaN in DSR calculation at step {self.current_step}. A_tm1={A_tm1}, B_tm1={B_tm1}, portfolio_return={actual_portfolio_return}. Using DSR=0."
            )
            dsr = 0.0
            # Consider adding more detailed logging or specific handling here.

        return dsr * self.reward_scaling

    def render(self, mode="human"):
        pass  # TODO: Implement visualization

    def close(self):
        pass
