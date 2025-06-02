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
    drl_config: DRLConfig,
    year_start: int,
    num_train_years: int,
    num_val_years: int,
    num_test_years: int,
    df_prices: pd.DataFrame,
    df_ret: pd.DataFrame,
    vol_df: pd.DataFrame,
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    """
    Slice data for a given window configuration with non-consecutive dates.

    Returns:
        Tuple of (train_data, val_data, test_data) where each is a tuple of (prices, returns, vol)
    """

    window_size = drl_config.env_window_size

    # Generate start and end dates
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

    def get_effective_start(df, start_date, window_size):
        """Find the true starting timestamp that is `window_size` rows before start_date."""
        date_index = df.index
        start_idx = date_index.searchsorted(start_date)
        return date_index[start_idx - window_size]

    # Compute effective starts
    train_start_eff = get_effective_start(df_prices, train_start_date, window_size)
    val_start_eff = get_effective_start(df_prices, val_start_date, window_size)
    test_start_eff = get_effective_start(df_prices, test_start_date, window_size)

    # Slicing
    train_prices = df_prices.loc[train_start_eff:train_end_date]
    train_returns = df_ret.loc[train_start_eff:train_end_date]
    train_vola = vol_df.loc[train_start_eff:train_end_date]

    val_prices = df_prices.loc[val_start_eff:val_end_date]
    val_returns = df_ret.loc[val_start_eff:val_end_date]
    val_vola = vol_df.loc[val_start_eff:val_end_date]

    test_prices = df_prices.loc[test_start_eff:test_end_date]
    test_returns = df_ret.loc[test_start_eff:test_end_date]
    test_vola = vol_df.loc[test_start_eff:test_end_date]

    # Sanity checks
    if train_prices.empty or val_prices.empty or test_prices.empty:
        raise ValueError(
            "One or more data slices are empty. Check date ranges and data availability."
        )

    return (
        (train_prices, train_returns, train_vola),
        (val_prices, val_returns, val_vola),
        (test_prices, test_returns, test_vola),
    )


def create_env_config(
    df_prices: pd.DataFrame,
    df_ret: pd.DataFrame,
    df_vola: pd.DataFrame,
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
    test_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
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
    test_prices, test_returns, test_vola = test_data

    # Create test environment
    env_test_config = create_env_config(
        test_prices, test_returns, test_vola, drl_config
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
    data_slices: Tuple[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ],
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
    train_data, val_data, test_data = data_slices
    train_prices, train_returns, train_vola = train_data
    val_prices, val_returns, val_vola = val_data

    # Check data length
    min_data_len = drl_config.env_window_size + 1
    if (
        len(train_prices) < min_data_len
        or len(val_prices) < min_data_len
        or len(test_data[0]) < min_data_len
    ):
        print(f"SKIPPING Window {window_idx + 1} due to insufficient data length.")
        return None, {"status": "skipped_insufficient_data", "metrics": {}}

    # Create environments
    env_train_config = create_env_config(
        train_prices, train_returns, train_vola, drl_config
    )
    env_val_config = create_env_config(val_prices, val_returns, val_vola, drl_config)

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
        best_agent_path, test_data, drl_config
    )

    return (
        best_agent_path,
        backtest_metrics,
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
    all_portfolios = {}

    os.makedirs(drl_config.model_save_dir, exist_ok=True)
    os.makedirs(drl_config.tensorboard_log_dir, exist_ok=True)

    drl_config.learning_rate_schedule = linear_schedule(
        drl_config.initial_lr, drl_config.final_lr
    )

    # Process each window
    for idx_window in range(drl_config.n_windows):
        current_start_year = drl_config.base_start_year + idx_window
        print(
            f"--- Starting Window {idx_window + 1}/{drl_config.n_windows} (Train Year Start: {current_start_year}) ---"
        )

        # Slice data for current window
        data_slices = slice_data(
            drl_config=drl_config,
            year_start=current_start_year,
            num_train_years=5,
            num_val_years=1,
            num_test_years=1,
            df_prices=df_prices,
            df_ret=df_ret,
            vol_df=df_vol,
        )

        # Process window
        previous_best_agent_path = (
            best_agent_paths_per_window[-1] if best_agent_paths_per_window else None
        )
        best_agent_path, window_results, backtest_portfolio = process_window(
            window_idx=idx_window,
            data_slices=data_slices,
            drl_config=drl_config,
            previous_best_agent_path=previous_best_agent_path,
        )

        best_agent_paths_per_window.append(best_agent_path)
        all_backtest_results.append(
            {
                "window": idx_window + 1,
                "best_agent_path": best_agent_path.split("/")[-1].split(".")[0],
                **window_results,
            }
        )
        all_portfolios[idx_window] = backtest_portfolio

    return all_backtest_results, all_portfolios
