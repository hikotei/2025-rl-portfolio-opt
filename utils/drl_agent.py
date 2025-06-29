import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Any


class DRLAgent:
    def __init__(
        self,
        env,
        n_envs: int = 1,
        policy: str = "MlpPolicy",
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.25,
        seed: int = 0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: str = "./logs/",
    ):
        """
        Initializes the DRLAgent.

        This constructor sets up the reinforcement learning environment and initializes
        the PPO (Proximal Policy Optimization) model with its specified configuration.

        Parameters:
        ----------
        env : object
            The environment instance. It should have attributes like `returns_df`,
            `prices_df`, `vola_df`, `window_size`, `transaction_cost`,
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
        learning_rate : float, optional
            The learning rate for the PPO optimizer. Default is 0.0003.
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
            "df_ret": env.df_ret,
            "df_prices": env.df_prices,
            "df_vola": env.df_vola,
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
        tb_experiment_name : str, optional
        """
        # Create a callback to log portfolio metrics
        callback = PortfolioLogCallback(self.env)

        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            tb_log_name=tb_experiment_name,
            callback=callback,
        )
        print(f"\nTraining complete. Trained for {total_timesteps} timesteps.")
        print(
            f"TensorBoard logs for experiment '{tb_experiment_name}' saved in directory: {self.model.tensorboard_log}"
        )

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        Predicts actions based on an observation.

        Using the trained PPO model to predict the next action(s)
        given the current observation(s) from the environment.
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
        """
        target_env = env if env is not None else self.env
        self.model = PPO.load(path, env=target_env)

    def evaluate(self, eval_env, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Evaluates the agent's performance on a given environment.

        This method runs the agent for a specified number of episodes on the
        evaluation environment (`eval_env`). It collects rewards and uses the
        environment's portfolio instance to calculate performance metrics.

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
            -   Other metrics calculated by the Portfolio's calc_metrics
                method based on the portfolio values of the first episode.
        """
        all_episode_rewards = []

        for episode in range(n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_total_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_total_reward += reward

            all_episode_rewards.append(episode_total_reward)

        mean_reward = (
            np.mean(all_episode_rewards) if len(all_episode_rewards) > 0 else np.nan
        )
        std_reward = (
            np.std(all_episode_rewards) if len(all_episode_rewards) > 0 else np.nan
        )

        eval_metrics = {
            "n_eval_episodes": n_eval_episodes,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }

        if n_eval_episodes > 0:
            # Get the portfolio instance from the environment
            portfolio = eval_env.portfolio if hasattr(eval_env, "portfolio") else None

            if portfolio is not None:
                # Use the portfolio's metrics directly
                portfolio_metrics = portfolio.calc_metrics()
                eval_metrics.update(portfolio_metrics)
                eval_metrics["final_portfolio_value_first_episode"] = round(
                    portfolio.current_balance
                )

        return eval_metrics, portfolio


class PortfolioLogCallback(BaseCallback):
    def __init__(self, train_env: VecEnv, verbose: int = 0):
        super(PortfolioLogCallback, self).__init__(verbose)
        self.train_env = train_env

    def _on_step(self) -> bool:
        if self.n_calls % self.model.n_steps == 0:  # Log at the same frequency as other logs
            # Get portfolio from the first environment (assuming all are similar)
            # Access the underlying environment of the VecEnv
            portfolios = self.train_env.get_attr("portfolio")
            if portfolios and len(portfolios) > 0:
                portfolio = portfolios[0] # Using the first env's portfolio for logging
                if portfolio and hasattr(portfolio, 'calc_metrics') and hasattr(portfolio, 'history') and len(portfolio.history) > 0:
                    metrics = portfolio.calc_metrics()
                    for key, value in metrics.items():
                        # Ensure value is a scalar and not NaN or Inf before logging
                        if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                            self.logger.record(f"portfolio/{key.replace(' ', '_').lower()}", value)
                        elif isinstance(value, np.number) and not (np.isnan(value) or np.isinf(value)): # Handles numpy numeric types
                            self.logger.record(f"portfolio/{key.replace(' ', '_').lower()}", float(value))
        return True
