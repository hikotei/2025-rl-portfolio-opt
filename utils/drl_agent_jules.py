import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from typing import Optional, Dict, Union, List, Any, Callable


class DRLAgent:
    def __init__(
        self,
        env,
        n_envs: int = 1,
        model_name: str = "ppo",
        policy: str = "MlpPolicy",
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        learning_rate: Union[float, Callable[[float], float]] = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.25,
        seed: int = 0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: str = "./tensorboard_logs/",
    ):
        """
        Initializes the DRLAgent.

        This constructor sets up the reinforcement learning environment and initializes
        the PPO (Proximal Policy Optimization) model with its specified configuration.

        Parameters:
        ----------
        env : object
            The environment instance. It should have attributes like `returns_df`,
            `prices_df`, `vol_df`, `window_size`, `transaction_cost`,
            `initial_balance`, and `reward_scaling`.
        n_envs : int, optional
            The number of parallel environments to use for training. Default is 1.
        model_name : str, optional
            The name of the DRL model to use. Currently, only 'ppo' is supported.
            Default is "ppo".
        policy : str, optional
            The policy network type. For PPO, common choices are "MlpPolicy" for
            multi-layer perceptron policies. Default is "MlpPolicy".
        n_steps : int, optional
            The number of steps to run for each environment per update. This is a PPO
            hyperparameter. Default is 2048.
        batch_size : int, optional
            The mini-batch size for PPO updates. Default is 64.
        n_epochs : int, optional
            The number of epochs to train the PPO model on the collected data per
            update. Default is 10.
        learning_rate : Union[float, Callable[[float], float]], optional
            The learning rate for the PPO optimizer. Can be a float or a schedule
            function. Default is 0.0003.
        gamma : float, optional
            The discount factor for future rewards. Default is 0.99.
        gae_lambda : float, optional
            Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            (GAE). Default is 0.95.
        clip_range : float, optional
            The clipping parameter for PPO, defining the range within which the policy
            ratio is clipped. Default is 0.25.
        seed : int, optional
            Random seed for reproducibility. Default is 0.
        policy_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments to pass to the policy network constructor.
            If None, a default configuration (Tanh activation, [64, 64] net arch,
            log_std_init=-1.0) is used. Default is None.

        Initializes:
        ------------
        -   `self.original_env_class`: Stores the class of the provided environment.
        -   `self.original_env_kwargs`: Stores the parameters of the original environment.
        -   `self.env`: A vectorized environment (SubprocVecEnv) for parallel training.
        -   `self.model`: The PPO model instance from stable-baselines3, configured
            with the specified hyperparameters.
        -   `self.training_metrics`: Initialized to None, will store training metrics later.
        """
        self.original_env_class = type(env)
        env_params = {
            "returns_df": env.returns_df,
            "prices_df": env.prices_df,
            "vol_df": env.vol_df,
            "window_size": env.window_size,
            "transaction_cost": env.transaction_cost,
            "initial_balance": env.initial_balance,
            "reward_scaling": env.reward_scaling,
        }
        if hasattr(env, "eta"):
            env_params["eta"] = env.eta
        self.original_env_kwargs = env_params

        def make_env_closure(rank: int, seed: int = 0, env_class=None, env_kwargs=None):
            def _init():
                if env_class is None:
                    raise ValueError("env_class must be provided to make_env_closure")
                current_env_kwargs = {**env_kwargs}
                new_env = env_class(**current_env_kwargs)
                new_env.reset(seed=seed + rank)
                return new_env

            set_random_seed(seed)
            return _init

        self.env = SubprocVecEnv(
            [
                make_env_closure(
                    i, seed, self.original_env_class, self.original_env_kwargs
                )
                for i in range(n_envs)
            ],
            start_method="fork",
        )

        if policy_kwargs is None:
            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh, net_arch=[64, 64], log_std_init=-1.0
            )

        if "log_std_init" in policy_kwargs:
            policy_kwargs["log_std_init"] = float(policy_kwargs["log_std_init"])

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
            policy_kwargs=policy_kwargs,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )
        self.training_metrics = None

    def train(self, total_timesteps: int = 100_000, tb_experiment_name: str = "ppo"):
        """
        Trains the PPO model.

        This method initiates the training process for the PPO model using the
        specified number of timesteps. It also handles TensorBoard logging for
        monitoring training progress.

        Parameters:
        ----------
        total_timesteps : int, optional
            The total number of samples (env steps) to train on. Default is 100,000.
        tb_experiment_name : str, optional
            The name of the experiment for TensorBoard logging. Default is "ppo".
        tensorboard_log_path : str, optional
            The path to the directory where TensorBoard logs will be saved.
            Default is "./tensorboard_logs/".
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            tb_log_name=tb_experiment_name,
        )
        print(f"\nTraining complete. Trained for {total_timesteps} timesteps.")
        print(
            f"TensorBoard logs for experiment '{tb_experiment_name}' saved in directory: {self.model.tensorboard_log}"
        )

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        Predicts actions based on an observation.

        This method uses the trained PPO model to predict the next action(s)
        given the current observation(s) from the environment.

        Parameters:
        ----------
        obs : np.ndarray
            The observation from the environment. This should be a NumPy array
            compatible with the observation space of the environment used for
            training.
        deterministic : bool, optional
            Whether to use deterministic or stochastic actions. If True, the model
            will output the action with the highest probability. If False, the
            action is sampled from the policy distribution. Default is True.

        Returns:
        -------
        np.ndarray
            The predicted action(s).
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        """
        Saves the trained PPO model.

        This method saves the current state of the PPO model to a file at the
        specified path. This allows for later reloading and reuse of the trained
        model.

        Parameters:
        ----------
        path : str
            The file path where the model should be saved. Typically, this is a
            `.zip` file for stable-baselines3 models.
        """
        self.model.save(path)

    def load(self, path: str, env=None):
        """
        Loads a pre-trained PPO model.

        This method loads a PPO model from a file specified by the path.
        It allows for setting a new environment for the loaded model or
        continuing with the agent's current environment.

        Parameters:
        ----------
        path : str
            The file path from which the model should be loaded. Typically, this
            is a `.zip` file saved by stable-baselines3.
        env : object, optional
            The environment to associate with the loaded model. If None, the
            agent's current environment (`self.env`) is used. This is useful
            if you want to load a model and use it with a different environment
            (e.g., an evaluation environment). Default is None.
        """
        target_env = env if env is not None else self.env
        self.model = PPO.load(path, env=target_env)

    def load_from_file(self, path: str, env=None):
        """
        Loads a model from a file and prints a confirmation message.

        This method is a convenience wrapper around the `load` method. It calls
        `self.load(path, env=env)` and then prints a message indicating that
        the model has been loaded from the specified path.

        Parameters:
        ----------
        path : str
            The file path from which the model should be loaded.
        env : object, optional
            The environment to associate with the loaded model. If None, the
            agent's current environment is used. Default is None.
        """
        self.load(path, env=env)
        print(f"Model loaded from {path}")

    def evaluate(self, eval_env, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Evaluates the agent's performance on a given environment.

        This method runs the agent for a specified number of episodes on the
        evaluation environment (`eval_env`). It collects rewards and portfolio
        values to calculate various performance metrics.

        Parameters:
        ----------
        eval_env : object
            The environment on which to evaluate the agent. This should be
            compatible with the agent's model (i.e., have the same observation
            and action spaces).
        n_eval_episodes : int, optional
            The number of episodes to run for evaluation. Default is 1.
        deterministic : bool, optional
            Whether to use deterministic actions during evaluation. If True, the
            model selects the action with the highest probability. If False,
            actions are sampled. Default is True.

        Returns:
        -------
        Dict[str, Any | float]
            A dictionary containing various evaluation metrics. This includes:
            -   `mean_reward`: Average reward per episode.
            -   `std_reward`: Standard deviation of rewards per episode.
            -   `n_eval_episodes`: Number of evaluation episodes.
            -   `final_portfolio_value_first_episode`: Portfolio value at the end
                of the first evaluation episode.
            -   Other metrics calculated by `self.calc_metrics()` based on the
                portfolio values of the first episode (e.g., Sharpe ratio,
                Max drawdown). If `n_eval_episodes` is 0, or if portfolio
                values cannot be extracted, these metrics might be NaN or default
                values.
        """
        all_episode_rewards = []
        all_episode_portfolio_values = []
        all_episode_weights_history = []

        for episode in range(n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_total_reward = 0.0
            current_episode_portfolio_values = [
                getattr(eval_env, "initial_balance", 100000)
            ]
            current_episode_weights_history = []
            if hasattr(eval_env, "get_current_weights"):
                current_episode_weights_history.append(eval_env.get_current_weights())
            # else: pass # Assuming PortfolioEnv, get_current_weights should exist.

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_total_reward += reward

                portfolio_val = None
                if isinstance(info, list) and len(info) > 0:
                    portfolio_val = info[0].get("portfolio_value")
                elif isinstance(info, dict):
                    portfolio_val = info.get("portfolio_value")

                if portfolio_val is not None:
                    current_episode_portfolio_values.append(portfolio_val)
                else:
                    current_episode_portfolio_values.append(
                        current_episode_portfolio_values[-1]
                        + reward  # Fallback, less accurate
                    )

                if hasattr(eval_env, "get_current_weights"):
                    current_episode_weights_history.append(
                        eval_env.get_current_weights()
                    )

            all_episode_rewards.append(episode_total_reward)
            all_episode_portfolio_values.append(current_episode_portfolio_values)
            all_episode_weights_history.append(current_episode_weights_history)

        mean_reward = (
            np.mean(all_episode_rewards) if len(all_episode_rewards) > 0 else np.nan
        )
        std_reward = (
            np.std(all_episode_rewards) if len(all_episode_rewards) > 0 else np.nan
        )

        if (
            n_eval_episodes > 0
            and len(all_episode_portfolio_values) > 0
            and len(all_episode_portfolio_values[0]) > 1
        ):
            eval_metrics = self.calc_metrics(all_episode_portfolio_values[0])
            final_portval_ep1 = all_episode_portfolio_values[0][-1]

            eval_metrics = {
                "n_eval_episodes": n_eval_episodes,
                "final_portfolio_value_first_episode": final_portval_ep1,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                **eval_metrics,
            }

        return eval_metrics

    def calc_metrics(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any | float]:
        """
        Calculates various financial performance metrics from a series of portfolio values.

        This method takes a time series of portfolio values and computes several
        standard financial metrics, such as annual return, volatility, Sharpe ratio,
        Sortino ratio, maximum drawdown, etc.

        Parameters:
        ----------
        portfolio_values : Union[List[float], np.ndarray, pd.Series]
            A list, NumPy array, or Pandas Series representing the portfolio value
            at each time step. Must contain at least two data points for meaningful
            calculations.
        risk_free_rate : float, optional
            The annualized risk-free rate used for calculating metrics like the
            Sharpe ratio and Sortino ratio. Default is 0.02 (2%).

        Returns:
        -------
        Dict[str, Any | float]
            A dictionary where keys are metric names (e.g., "Annual return",
            "Sharpe ratio") and values are the calculated metric values.
            If input `portfolio_values` is invalid (e.g., too short, contains NaN/Inf,
            all zeros), it returns a dictionary with default NaN values for metrics
            and an "error" key explaining the issue.
        """
        default_metrics = {
            "Annual return": np.nan,
            "Cumulative returns": np.nan,
            "Annual volatility": np.nan,
            "Sharpe ratio": np.nan,
            "Calmar ratio": np.nan,
            "Stability": np.nan,
            "Max drawdown": np.nan,
            "Omega ratio": np.nan,
            "Sortino ratio": np.nan,
            "Skew": np.nan,
            "Kurtosis": np.nan,
            "Tail ratio": np.nan,
            "Daily value at risk (95%)": np.nan,
            "Portfolio turnover": np.nan,
        }
        if (
            not isinstance(portfolio_values, (list, np.ndarray, pd.Series))
            or len(portfolio_values) < 2
        ):
            return {
                **default_metrics,
                "error": "Portfolio values must be a list/array with at least 2 elements.",
            }

        portfolio_values = np.array(portfolio_values, dtype=float)

        if np.any(np.isnan(portfolio_values)) or np.any(np.isinf(portfolio_values)):
            return {**default_metrics, "error": "Portfolio values contain NaN or Inf."}

        # Attempt to handle initial zero portfolio value if it recovers
        if portfolio_values[0] == 0:
            non_zero_indices = np.where(portfolio_values != 0)[0]
            if len(non_zero_indices) > 0:
                first_non_zero_idx = non_zero_indices[0]
                if (
                    first_non_zero_idx < len(portfolio_values) - 1
                ):  # Need at least 2 points after this
                    portfolio_values = portfolio_values[first_non_zero_idx:]
                else:  # Not enough data after removing initial zeros
                    return {
                        **default_metrics,
                        "error": "Not enough data after removing initial zero(s).",
                    }
            else:  # All values are zero
                return {**default_metrics, "error": "All portfolio values are zero."}

        if len(portfolio_values) < 2:  # Check again after potential slicing
            return {
                **default_metrics,
                "error": "Not enough data points after processing initial values.",
            }

        denominator = portfolio_values[:-1].copy()
        valid_denominator_mask = (
            denominator > 1e-9
        )  # Check for positive, non-tiny values

        daily_returns = np.full_like(denominator, np.nan)
        if np.any(valid_denominator_mask):
            diff_values = np.diff(portfolio_values)
            daily_returns[valid_denominator_mask] = (
                diff_values[valid_denominator_mask]
                / denominator[valid_denominator_mask]
            )

        daily_returns = daily_returns[~np.isnan(daily_returns)]

        if len(daily_returns) == 0:
            return {
                **default_metrics,
                "error": "No valid daily returns could be calculated.",
            }

        pv_current_initial = portfolio_values[
            0
        ]  # This is the first value used for calculation after any slicing
        pv_final = portfolio_values[-1]

        # Use pv_initial_original for cumulative return if it makes sense, or pv_current_initial
        # For consistency, using pv_current_initial as the basis for returns calc after data cleaning
        annual_return = (
            (pv_final / pv_current_initial) ** (252 / len(daily_returns)) - 1
            if pv_current_initial != 0
            else np.nan
        )
        cumulative_return = (
            (pv_final / pv_current_initial) - 1 if pv_current_initial != 0 else np.nan
        )

        annual_volatility = np.std(daily_returns) * np.sqrt(252)

        daily_risk_free_rate = risk_free_rate / 252
        excess_returns = daily_returns - daily_risk_free_rate
        std_dev_returns = np.std(daily_returns)

        sharpe_ratio = (
            np.mean(excess_returns) / std_dev_returns * np.sqrt(252)
            if std_dev_returns > 1e-9
            else np.nan
        )

        rolling_max = np.maximum.accumulate(
            portfolio_values
        )  # Use full original values for drawdown context if possible
        # Or stick to current `portfolio_values` slice
        drawdowns = (
            portfolio_values - rolling_max
        ) / rolling_max  # Max drawdown based on current slice
        max_drawdown = (
            np.min(drawdowns) if len(drawdowns) > 0 and np.any(rolling_max > 0) else 0.0
        )

        calmar_ratio = (
            annual_return / abs(max_drawdown)
            if abs(max_drawdown) > 1e-9 and pd.notna(annual_return)
            else np.nan
        )

        negative_returns = daily_returns[daily_returns < daily_risk_free_rate]
        if len(negative_returns) > 0:
            downside_std_dev = np.std(negative_returns)
            sortino_ratio = (
                np.mean(excess_returns) / downside_std_dev * np.sqrt(252)
                if downside_std_dev > 1e-9
                else np.nan
            )
        else:
            mean_er = np.mean(excess_returns)
            sortino_ratio = (
                np.inf if mean_er > 1e-9 else (0 if abs(mean_er) < 1e-9 else np.nan)
            )

        threshold = daily_risk_free_rate
        gains = daily_returns[daily_returns > threshold] - threshold
        losses = daily_returns[daily_returns <= threshold] - threshold

        sum_gains = np.sum(gains)
        sum_abs_losses = abs(np.sum(losses))

        if sum_abs_losses < 1e-9:
            omega_ratio = (
                np.inf if sum_gains > 1e-9 else (1 if abs(sum_gains) < 1e-9 else np.nan)
            )
        else:
            omega_ratio = sum_gains / sum_abs_losses

        skew = pd.Series(daily_returns).skew()
        kurtosis = pd.Series(daily_returns).kurtosis()

        tail_ratio = np.nan
        var_95 = np.nan
        if len(daily_returns) >= 20:
            percentile_5 = np.percentile(daily_returns, 5)
            percentile_95 = np.percentile(daily_returns, 95)
            var_95 = percentile_5
            if abs(percentile_5) < 1e-9:
                tail_ratio = (
                    np.inf
                    if percentile_95 > 1e-9
                    else (1 if abs(percentile_95) < 1e-9 else np.nan)
                )
            else:
                tail_ratio = percentile_95 / abs(percentile_5)

        portfolio_turnover = np.nan
        stability = (
            1 / (1 + annual_volatility) if pd.notna(annual_volatility) else np.nan
        )

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
            "Daily value at risk (95%)": var_95,
            "Portfolio turnover": portfolio_turnover,
        }
