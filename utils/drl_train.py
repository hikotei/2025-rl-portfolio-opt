import os
import pandas as pd
import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, Any

from utils.drl_agent import DRLAgent
from utils.config import DRLConfig
from utils.portfolio_env import PortfolioEnv


def linear_schedule(
    initial_value: float, final_value: float
) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: schedule that computes current learning rate depending on progress remaining (1.0 -> 0.0)
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1.0 to 0.0
        """
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def dataload(
    price_data_path: str, returns_data_path: str, vola_data_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate the required datasets.

    Args:
        price_data_path: Path to price data parquet file
        returns_data_path: Path to returns data parquet file
        vola_data_path: Path to volatility data parquet file

    Returns:
        Tuple of (df_prices, df_ret, df_vol)
    """
    try:
        print("Loading data...")
        df_prices = pd.read_parquet(price_data_path)
        df_ret = pd.read_parquet(returns_data_path)
        df_vol = pd.read_parquet(vola_data_path)
        print("Data loaded successfully.")

        # Ensure DataFrames have DateTimeIndex
        for df in [df_prices, df_ret, df_vol]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        return df_prices, df_ret, df_vol

    except FileNotFoundError as e:
        print(f"ERROR: Data file not found. {e}")
        print("Please ensure data is generated and paths are correct.")
        raise


def slice_data(
    year_start: int,
    num_train_years: int,
    num_val_years: int,
    num_test_years: int,
):
    """
    Args:
        year_start: Starting year for the window
        num_train_years: Number of years for training
        num_val_years: Number of years for validation
        num_test_years: Number of years for testing
    """
    train_start_date = pd.to_datetime(f"{year_start}-01-01")
    train_end_date = pd.to_datetime(f"{year_start + num_train_years - 1}-12-31")

    val_start_date = pd.to_datetime(f"{year_start + num_train_years}-01-01")
    val_end_date = pd.to_datetime(
        f"{year_start + num_train_years + num_val_years - 1}-12-31"
    )

    test_start_date = pd.to_datetime(
        f"{year_start + num_train_years + num_val_years}-01-01"
    )
    test_end_date = pd.to_datetime(
        f"{year_start + num_train_years + num_val_years + num_test_years - 1}-12-31"
    )

    print(f"  Train Period: {train_start_date.date()} to {train_end_date.date()}")
    print(f"  Val Period  : {val_start_date.date()} to {val_end_date.date()}")
    print(f"  Test Period : {test_start_date.date()} to {test_end_date.date()}")

    # return just the dates
    return (
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date,
    )


def create_env_config(
    df_prices: pd.DataFrame,
    df_ret: pd.DataFrame,
    df_vola: pd.DataFrame,
    start_date,
    end_date,
    drl_config: DRLConfig,
) -> Dict[str, Any]:
    """
    Create environment configuration dictionary.

    Args:
        df_prices: Price dataframe
        df_ret: Returns dataframe
        vol_df: Volatility dataframe
        drl_config: Training configuration

    Returns:
        Dictionary of environment parameters
    """
    return {
        "returns_df": df_ret,
        "prices_df": df_prices,
        "vola_df": df_vola,
        "window_size": drl_config.env_window_size,
        "transaction_cost": drl_config.transaction_cost,
        "initial_balance": drl_config.initial_balance,
        "reward_scaling": drl_config.reward_scaling,
        "eta": drl_config.eta_dsr,
        "start_date": start_date,
        "end_date": end_date,
    }


def train_single_agent(
    env_train: PortfolioEnv,
    env_val: PortfolioEnv,
    drl_config: DRLConfig,
    agent_seed: int,
    previous_best_agent_path: Optional[str] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Train a single agent and return its metrics and path.

    Args:
        env_train: Training environment
        env_val: Validation environment
        drl_config: Training configuration
        agent_seed: Random seed for the agent
        previous_best_agent_path: Path to previous best agent for seeding

    Returns:
        Tuple of (validation metrics, agent save path)
    """
    # Create agent
    agent = DRLAgent(
        env=env_train,
        n_envs=drl_config.n_envs,
        policy_kwargs=drl_config.policy_kwargs,
        n_steps=drl_config.n_steps_per_env,
        batch_size=drl_config.batch_size,
        n_epochs=drl_config.n_epochs,
        learning_rate=drl_config.learning_rate_schedule,
        gamma=drl_config.gamma,
        gae_lambda=drl_config.gae_lambda,
        clip_range=drl_config.clip_range,
        seed=agent_seed,
        tensorboard_log=drl_config.tensorboard_log_dir,
    )

    # Load previous best agent if available
    if previous_best_agent_path is not None:
        print(f"    Seeding agent from: {previous_best_agent_path}")
        agent.load(path=previous_best_agent_path, env=None)
        agent.model.set_random_seed(agent_seed)

    # Train agent
    print(
        f"    Starting training for {drl_config.total_timesteps_per_round} timesteps..."
    )
    agent.train(
        total_timesteps=drl_config.total_timesteps_per_round,
        tb_experiment_name=f"PPO_Seed{agent_seed}",
    )

    # Evaluate agent
    print("    Evaluating agent on validation set...")
    val_metrics, val_portfolio = agent.evaluate(eval_env=env_val, n_eval_episodes=1)
    current_val_reward = val_metrics.get("mean_reward", -np.inf)
    print(f"    Validation Mean Reward: {current_val_reward:.4f}")

    # Save agent
    current_agent_model_name = (
        f"agent_seed{agent_seed}_valrew{current_val_reward:.2f}.zip"
    )
    current_agent_save_path = os.path.join(
        drl_config.model_save_dir, current_agent_model_name
    )
    agent.save(current_agent_save_path)
    print(f"    Agent saved to: {current_agent_save_path}")

    return val_metrics, current_agent_save_path


def backtest_agent(
    agent_path: str,
    df_prices: pd.DataFrame,
    df_ret: pd.DataFrame,
    df_vol: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    drl_config: DRLConfig,
) -> Dict[str, float]:
    """
    Run backtest for a single agent.

    Args:
        agent_path: Path to the trained agent
        test_data: Tuple of (test_prices, test_returns, test_vola)
        drl_config: Training configuration

    Returns:
        Dictionary of backtest metrics
    """
    # Create test environment
    env_test_config = create_env_config(
        df_prices, df_ret, df_vol, start_date, end_date, drl_config
    )
    env_test = PortfolioEnv(**env_test_config)

    # Load agent
    agent = DRLAgent(
        env=env_test,
        n_envs=1,
        policy_kwargs=drl_config.policy_kwargs,
    )
    agent.load(path=agent_path, env=env_test)

    # Run backtest
    print("    Running backtest evaluation...")
    backtest_metrics, backtest_portfolio = agent.evaluate(
        eval_env=env_test, n_eval_episodes=1
    )

    return backtest_metrics, backtest_portfolio


def process_window(
    window_idx: int,
    date_slices: Tuple,
    df_prices: pd.DataFrame,
    df_ret: pd.DataFrame,
    df_vol: pd.DataFrame,
    drl_config: DRLConfig,
    previous_best_agent_path: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Process a single training window.

    Args:
        window_idx: Index of the current window
        data_slices: Tuple of (train_data, val_data, test_data)
        drl_config: Training configuration
        previous_best_agent_path: Path to previous best agent for seeding

    Returns:
        Tuple of (best_agent_path, backtest_results)
    """

    (
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date,
    ) = date_slices

    # Create environments
    env_train_config = create_env_config(
        df_prices, df_ret, df_vol, train_start_date, train_end_date, drl_config
    )
    env_val_config = create_env_config(
        df_prices, df_ret, df_vol, val_start_date, val_end_date, drl_config
    )

    env_train = PortfolioEnv(**env_train_config)
    env_val = PortfolioEnv(**env_val_config)

    # Train multiple agents
    best_val_reward = -np.inf
    best_agent_path = None

    for i_agent in range(drl_config.agents_per_window):

        agent_seed = (window_idx * drl_config.agents_per_window) + i_agent
        print(
            f"  Training Agent {i_agent + 1}/{drl_config.agents_per_window} with seed {agent_seed}..."
        )

        val_metrics, agent_path = train_single_agent(
            env_train=env_train,
            env_val=env_val,
            drl_config=drl_config,
            agent_seed=agent_seed,
            previous_best_agent_path=previous_best_agent_path,
        )

        current_val_reward = val_metrics.get("mean_reward", -np.inf)
        if current_val_reward > best_val_reward:
            best_val_reward = current_val_reward
            best_agent_path = agent_path

    if best_agent_path is None:
        return None, {"status": "no_best_agent", "metrics": {}}

    # Run backtest
    backtest_metrics, backtest_portfolio = backtest_agent(
        best_agent_path,
        df_prices,
        df_ret,
        df_vol,
        test_start_date,
        test_end_date,
        drl_config,
    )

    return (
        best_agent_path,
        {"status": "completed", "metrics": backtest_metrics},
        backtest_portfolio,
    )


def training_pipeline(
    drl_config: DRLConfig, df_prices, df_ret, df_vol
) -> List[Dict[str, Any]]:
    """
    Main training pipeline that orchestrates the entire process.

    Args:
        drl_config: Training configuration
        data_paths: Data file paths configuration

    Returns:
        List of backtest results for each window
    """

    all_backtest_results = []
    best_agent_paths_per_window = []
    all_portfolios = []

    os.makedirs(drl_config.model_save_dir, exist_ok=True)
    os.makedirs(drl_config.tensorboard_log_dir, exist_ok=True)

    drl_config.learning_rate_schedule = linear_schedule(
        drl_config.initial_lr, drl_config.final_lr
    )

    # Process each window
    for i_window in range(drl_config.n_windows):
        current_start_year = drl_config.base_start_year + i_window
        print(
            f"--- Starting Window {i_window + 1}/{drl_config.n_windows} (Train Year Start: {current_start_year}) ---"
        )

        # Slice data for current window
        date_slices = slice_data(
            year_start=current_start_year,
            num_train_years=5,
            num_val_years=1,
            num_test_years=1,
        )

        # Process window
        previous_best_agent_path = (
            best_agent_paths_per_window[-1] if best_agent_paths_per_window else None
        )
        best_agent_path, window_results, backtest_portfolio = process_window(
            window_idx=i_window,
            date_slices=date_slices,
            df_prices=df_prices,
            df_ret=df_ret,
            df_vol=df_vol,
            drl_config=drl_config,
            previous_best_agent_path=previous_best_agent_path,
        )

        best_agent_paths_per_window.append(best_agent_path)
        all_backtest_results.append(
            {
                "window": i_window + 1,
                "best_agent_path": best_agent_path,
                **window_results,
            }
        )
        all_portfolios.append(backtest_portfolio)

    return all_backtest_results, all_portfolios
