import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from typing import Dict, Tuple, Optional, Union, List
from utils.portfolio import Portfolio


class PortfolioEnv(gym.Env):
    """
    A Gymnasium-compatible environment for portfolio optimization
    using Deep Reinforcement Learning (DRL).

    This environment simulates a trading scenario where an agent decides how to
    allocate money across different assets to maximize risk adjusted returns.

    The reward is the Differential Sharpe Ratio (DSR).

    State space includes:
    - Historical returns matrix
    - Volatility features (20-day vol, VIX, vol ratio)
    - Current portfolio weights + cash weight

    Action space:
    - Continuous portfolio weights (sum to 1)
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        vola_df: pd.DataFrame,
        window_size: int = 60,
        transaction_cost: float = 0,
        initial_balance: float = 100_000,
        reward_scaling: float = 1.0,
        eta: float = 1 / 252,  # smooth param for DSR
    ):
        """
        Initialize the portfolio environment with historical data and parameters.

        prices_df is used for step function rebalancing of weights and portfolio value calculation
        returns_df and vola_df are only used to construct observations
        """
        super().__init__()

        # Store data
        self.returns_df = returns_df
        self.prices_df = prices_df
        self.vola_df = vola_df
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling

        # Exponential moving average
        self.eta = eta
        self.A_t = 0.0  # EMA of returns
        self.B_t = 0.0  # EMA of squared returns
        self.prev_A_t = 0.0
        self.prev_B_t = 0.0
        self.prev_sharpe = 0.0

        # Identify columns with NaN values
        self.nan_cols = []
        for col in returns_df.columns:
            if returns_df[col].isna().any() or prices_df[col].isna().any():
                self.nan_cols.append(col)

        # Get number of assets (including those with NaNs)
        self.n_assets = len(returns_df.columns)

        # UserWarning: We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) 
        # cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

        # Define action space (portfolio weights)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Define observation space as outlined in paper
        # TODO: think about flattening the observation matrix ...
        # UserWarning: Your observation  has an unconventional shape (neither an image, nor a 1D vector). 
        # We recommend you to flatten the observation to have only a 1D vector or use a custom policy to properly process the data.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets + 1, 1 + self.window_size),  # (rows, cols)
            dtype=np.float32,
        )

        # Initialize portfolio
        self.portfolio = Portfolio(returns_df.columns, initial_balance)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.

        This is like starting a new trading day with a fresh portfolio.
        For example, if you're training the agent, you might want to start over
        multiple times to learn from different market conditions.

        Returns:
            observation, info
        """
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio.reset()
        self.A_t = 0.0
        self.B_t = 0.0
        self.prev_A_t = 0.0
        self.prev_B_t = 0.0
        self.prev_sharpe = 0.0
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one step/day in the environment by executing a trading action.

        Args:
            action: Array of portfolio weights (e.g., [0.4, 0.3, 0.3] for three stocks)
            should already be correctly normalized to sum to 1

        Returns:
            observation, reward, terminated, truncated, info
        """
        # # Handle vectorized actions - each environment should get its corresponding action
        # if len(action.shape) == 2:
        #     # Get the action for this specific environment
        #     action = action[0]  # This environment's action

        # # Convert single float to array if needed
        # if isinstance(action, (np.float32, float)):
        #     action = np.array([action] * self.n_assets, dtype=np.float32)

        # Check for NaN in action
        if np.isnan(action).any():
            raise ValueError("NaN values in action")

        # Convert action to weights using softmax to ensure they sum to 1
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)

        # Handle NaN columns - set weights to 0 for assets with missing data
        mask = self.prices_df.columns.isin(self.nan_cols)
        weights[mask] = 0

        # Get current prices
        current_prices = self.prices_df.iloc[self.current_step].fillna(0)

        # Rebalance portfolio with new weights
        self.portfolio.update_rebalance(
            current_prices,
            dict(zip(self.prices_df.columns, weights)),
            date=self.prices_df.index[self.current_step],
        )

        # Calculate portfolio return
        portfolio_return = self.portfolio.get_return()

        # Check for NaN in portfolio return
        if np.isnan(portfolio_return):
            raise ValueError(f"NaN in portfolio return at step {self.current_step}")

        # Update DSR state before updating A_t and B_t
        reward = self._calculate_reward(portfolio_return)

        # Update A_t and B_t for next step
        self.prev_A_t = self.A_t
        self.prev_B_t = self.B_t
        self.A_t = self.prev_A_t + self.eta * (portfolio_return - self.prev_A_t)
        self.B_t = self.prev_B_t + self.eta * (portfolio_return**2 - self.prev_B_t)

        # Check for NaN in moving averages
        if np.isnan(self.A_t) or np.isnan(self.B_t):
            raise ValueError(f"NaN in moving averages at step {self.current_step}")

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.returns_df) - 1
        truncated = False

        # Get new observation
        observation = self._get_observation()

        info = {
            "portfolio_value": self.portfolio.current_balance,
            "portfolio_return": portfolio_return,
            "shares": self.portfolio.positions,
            "weights": self.portfolio.weights,
            "w_c": self.portfolio.w_c,
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Construct the current state observation that the agent sees.
        Returns:
            State observation array
        """
        # Get returns window (shape: window_size x n_assets)
        returns_window = self.returns_df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        returns_window = np.nan_to_num(returns_window, nan=0.0)

        # Get current weights from portfolio
        weights_full = np.append(
            list(self.portfolio.weights.values()), self.portfolio.w_c
        )

        # Prepare observation matrix: (n_assets+1) x (window_size + 4)
        n_assets = self.n_assets
        obs_rows = n_assets + 1
        obs_cols = 1 + self.window_size  # 1 for weights, window_size for returns
        observation = np.zeros((obs_rows, obs_cols), dtype=np.float32)

        # Fill first column with weights
        observation[:, 0] = weights_full

        # Fill next window_size columns with returns (cash row remains zero)
        observation[:n_assets, 1 : 1 + self.window_size] = returns_window.T

        # Fill 3 values after w_c in last row with vol features
        vol_features = self.vola_df.iloc[self.current_step].values  # shape: (3,)
        observation[-1, 1:4] = vol_features

        # Check for NaN values
        if np.isnan(observation).any():
            raise ValueError(f"NaN values in observation at step {self.current_step}")

        return observation

    def _calculate_reward(self, portfolio_return: float) -> float:
        """
        Calculate the reward for the agent's action using the Differential Sharpe Ratio (DSR).

        This is like giving the agent a score for its trading decision.
        Instead of just looking at returns, it considers both returns and risk.
        For example:
        - High returns with low risk = high reward
        - High returns with high risk = lower reward
        - Low returns with high risk = negative reward

        Args:
            portfolio_return: Current portfolio return

        Returns:
            Reward value
        """
        # If first step, no DSR can be computed
        if self.current_step == self.window_size:
            return 0.0

        # Use previous A_t and B_t for DSR calculation
        A_tm1 = self.prev_A_t
        B_tm1 = self.prev_B_t

        # Compute deltas
        delta_A = portfolio_return - A_tm1
        delta_B = portfolio_return**2 - B_tm1

        # Check for NaN in deltas
        if np.isnan(delta_A) or np.isnan(delta_B):
            raise ValueError(f"NaN in deltas at step {self.current_step}")

        # Differential Sharpe Ratio (Moody et al. 1998)
        num = B_tm1 * delta_A - 0.5 * A_tm1 * delta_B
        denom_dsr = (B_tm1 - A_tm1**2) ** 1.5

        # Add numerical stability
        if denom_dsr < 1e-8:
            denom_dsr = 1e-8

        dsr = num / denom_dsr

        # Check for NaN in DSR
        if np.isnan(dsr):
            raise ValueError(f"NaN in DSR at step {self.current_step}")

        return dsr * self.reward_scaling

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        this could be used to show a trading dashboard ...
        """
        pass

    def close(self):
        """
        Clean up environment resources.
        """
        pass

    def calc_metrics(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        risk_free_rate: float = 0.02,
    ):
        """
        Calculate various portfolio performance metrics.

        Args:
            portfolio_values: Array of portfolio values over time
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Dictionary containing various portfolio metrics
        """
        # Convert to numpy array if needed
        if isinstance(portfolio_values, (list, pd.Series)):
            portfolio_values = np.array(portfolio_values)

        # Calculate daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Annual return (assuming 252 trading days)
        annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (
            252 / len(portfolio_values)
        ) - 1

        # Cumulative returns
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        # Annual volatility
        annual_volatility = np.std(daily_returns) * np.sqrt(252)

        # Sharpe ratio
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(daily_returns)

        # Maximum drawdown
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio (using negative returns only)
        negative_returns = daily_returns[daily_returns < 0]
        sortino_ratio = (
            np.sqrt(252) * np.mean(excess_returns) / np.std(negative_returns)
            if len(negative_returns) > 0
            else 0
        )

        # Omega ratio
        threshold = risk_free_rate / 252
        positive_returns = daily_returns[daily_returns > threshold]
        negative_returns = daily_returns[daily_returns <= threshold]
        omega_ratio = (
            np.sum(positive_returns - threshold)
            / abs(np.sum(negative_returns - threshold))
            if len(negative_returns) > 0
            else float("inf")
        )

        # Skewness and Kurtosis
        skew = pd.Series(daily_returns).skew()
        kurtosis = pd.Series(daily_returns).kurtosis()

        # Tail ratio (95th percentile / 5th percentile)
        tail_ratio = np.percentile(daily_returns, 95) / abs(
            np.percentile(daily_returns, 5)
        )

        # Value at Risk (95%)
        var = np.percentile(daily_returns, 5)

        # Portfolio turnover (average daily change in weights)
        daily_change = np.mean(
            np.abs(np.diff(portfolio_values) / portfolio_values[:-1])
        )

        # Stability (inverse of volatility)
        stability = 1 / (1 + annual_volatility)

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
            "Portfolio turnover": daily_change,
        }
