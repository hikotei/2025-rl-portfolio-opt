import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Union, List, Any


class DRLAgent:
    """
    DRLAgent for portfolio optimization using PPO and gymnasium environments.
    """

    def __init__(
        self,
        env,
        n_envs: int = 1,
        model_name: str = "ppo",
        policy: str = "MlpPolicy",
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.25,
        seed: int = 0,
    ):
        """
        Initialize PPO agent with given environment and parameters.
        Args:
            env:
                The environment instance (should be Gym-compatible)
            n_envs:
                Number of environments to run in parallel (default: 1)
            model_name:
                Name of the model (default: 'ppo')
            policy:
                Policy type for PPO (default: 'MlpPolicy')
            n_steps:
                Number of steps to run for each environment per update (default: 2048)
            batch_size:
                Minibatch size for PPO (default: 64)
            n_epochs:
                Number of epochs when optimizing the surrogate loss (default: 10)
            learning_rate:
                Learning rate (default: 0.0003)
            gamma:
                Discount factor (default: 0.99)
            gae_lambda:
                Factor for trade-off of bias vs variance for GAE (default: 0.95)
            clip_range:
                Clipping parameter for PPO (default: 0.2)
            seed:
                Random seed for reproducibility (default: 0)
        """

        def make_env(rank: int, seed: int = 0):
            """
            Utility function for multiprocessed env.

            :param rank: (int) index of the subprocess
            :param seed: (int) the initial seed for RNG
            """

            def _init():
                # create new env from the same class as the given env
                new_env = type(env)(
                    returns_df=env.returns_df,
                    prices_df=env.prices_df,
                    vol_df=env.vol_df,
                    window_size=env.window_size,
                    transaction_cost=env.transaction_cost,
                    initial_balance=env.initial_balance,
                    reward_scaling=env.reward_scaling,
                    eta=env.eta,
                )
                # use a seed for reproducibility
                # Important: use a different seed for each environment
                # otherwise they would generate the same experiences
                new_env.reset(seed=seed+rank)
                return new_env

            set_random_seed(seed)
            return _init

        # Create vectorized environment with proper closures
        self.env = SubprocVecEnv(
            [make_env(i, seed) for i in range(n_envs)],
            start_method="fork"
        )

        self.model = PPO(
            policy,
            self.env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
        )
        self.training_metrics = None

    def train(self, total_timesteps: int = 100_000, seed: Optional[int] = None):
        """
        Train the PPO agent.
        Args:
            total_timesteps: Number of timesteps to train
            seed: Random seed for reproducibility
        """
        if seed is not None:
            self.model.set_random_seed(seed)

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
        )

        # Get the final portfolio value from the environment
        obs = self.env.reset()[0]
        done = False
        # Track portfolio values from first environment only
        portfolio_values = []
        # Track final portfolio values from all environments
        final_portfolio_values = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            # not that there is no truncated output !!!
            obs, reward, terminated, info = self.env.step(action)
            done = terminated.any()  # no truncated

            # Handle info as a list of dictionaries for multiple environments
            if isinstance(info, list):
                # Track portfolio values from first environment only
                if "portfolio_value" in info[0]:
                    portfolio_values.append(info[0]["portfolio_value"])
                # Store final portfolio values from all environments
                for env_info in info:
                    if "portfolio_value" in env_info:
                        final_portfolio_values.append(env_info["portfolio_value"])
            else:
                if "portfolio_value" in info:
                    portfolio_values.append(info["portfolio_value"])
                    final_portfolio_values.append(info["portfolio_value"])

        # Calculate and store training metrics
        if len(portfolio_values) > 0:
            # Calculate metrics using first environment's portfolio values
            self.training_metrics = self.calc_metrics(portfolio_values)

            # Print training summary
            print("\nTraining Summary:")
            print(f"Final Portfolio Value (First Env): ${portfolio_values[-1]:,.2f}")
            if len(final_portfolio_values) > 1:
                print(
                    f"Average Final Portfolio Value (All Envs): ${np.mean(final_portfolio_values):,.2f}"
                )
                print(
                    f"Std Final Portfolio Value (All Envs): ${np.std(final_portfolio_values):,.2f}"
                )
            print("\nPerformance Metrics (First Env):")
            for metric, value in self.training_metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        Get action from the trained agent.
        Args:
            obs: Observation/state
            deterministic: Whether to use deterministic actions
        Returns:
            Action selected by the agent
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        """
        Save the trained model to the given path.
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load a trained model from the given path.
        """
        self.model = PPO.load(path, env=self.env)

    def evaluate(self, env, n_episodes: int = 10):
        """
        Evaluate the agent on the environment with detailed metrics.
        Args:
            env: The environment to evaluate on
            n_episodes: Number of episodes to run
        Returns:
            Dictionary containing evaluation metrics
        """
        all_portfolio_values = []
        rewards = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            episode_portfolio_values = []

            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

                if "portfolio_value" in info:
                    episode_portfolio_values.append(info["portfolio_value"])

            rewards.append(total_reward)
            all_portfolio_values.extend(episode_portfolio_values)

        # Calculate evaluation metrics using agent's method
        eval_metrics = self.calc_metrics(all_portfolio_values)
        eval_metrics["Average Reward"] = np.mean(rewards)

        # Print evaluation summary
        print("\nEvaluation Summary:")
        print(f"Final Portfolio Value: ${all_portfolio_values[-1]:,.2f}")
        print(f"Average Reward: {np.mean(rewards):.4f}")
        print("\nPerformance Metrics:")
        for metric, value in eval_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        return eval_metrics

    def calc_metrics(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        weights_history: Optional[List[Dict[str, float]]] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any | float]:
        """
        Calculates a comprehensive set of portfolio performance metrics from a time series of portfolio values.
        
        Args:
            portfolio_values: Sequence of portfolio values over time.
            weights_history: Optional list of dictionaries with portfolio weights at each time step.
            risk_free_rate: Annual risk-free rate used in certain metrics (default: 2%).
        
        Returns:
            Dictionary containing annual return, cumulative returns, annual volatility, Sharpe ratio,
            Calmar ratio, stability, maximum drawdown, Omega ratio, Sortino ratio, skewness, kurtosis,
            tail ratio, daily value at risk (VaR), and portfolio turnover.
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
