import os
import pandas as pd
from typing import Dict, Any, List, Tuple

from utils.config import DRLConfig
from utils.drl_train import backtest_agent, slice_data


def find_best_agents(model_dir: str, n_windows: int) -> Tuple[List[str], Dict[int, List[float]]]:
    """
    For each window, find the agent file with the highest val reward in its filename.
    Also collect all validation rewards for each window.
    Assumes agent files are named like 'agent_{window}-{agent}_seed=..._test=..._valrew={val:.2f}.zip'
    """
    print(f"model_dir: {model_dir}")
    agent_files = [
        f
        for f in os.listdir(model_dir)
        if f.endswith(".zip") and f.startswith("agent_")
    ]
    window_best = {}
    window_rewards = {}
    
    for f in agent_files:
        window = int(f.split("agent_")[1].split("-")[0])
        val = float(f.split("_valrew=")[1].split(".zip")[0])
        if window not in window_best or val > window_best[window][0]:
            window_best[window] = (val, f)
        if window not in window_rewards:
            window_rewards[window] = []
        window_rewards[window].append(val)

    print(f"best agents: {window_best}")
    best_paths = [
        os.path.join(model_dir, window_best[w][1]) for w in sorted(window_best.keys())
    ]
    return best_paths, window_rewards


def evaluation_pipeline(
    drl_config: DRLConfig, df_prices, df_ret, df_vol
) -> Tuple[pd.DataFrame, Dict[int, Any]]:
    """
    Evaluate best agents in each window and save backtest portfolios and summary.
    Args:
        drl_config: Training configuration
        df_prices: Price data DataFrame
        df_ret: Returns data DataFrame
        df_vol: Volatility data DataFrame
    Returns:
        Tuple of (results_df, all_portfolios)
    """
    all_backtest_results = []
    all_portfolios = {}

    best_agent_paths, window_rewards = find_best_agents(drl_config.model_save_dir, drl_config.n_windows)

    for window_idx, agent_path in enumerate(best_agent_paths):
        current_start_year = drl_config.base_start_year + window_idx
        print(
            f"--- Evaluating Window {window_idx + 1}/{drl_config.n_windows} (Test Year Start: {current_start_year + 6}) ---"
        )
        # Slice data for current window
        _, _, test_data = slice_data(
            drl_config=drl_config,
            year_start=current_start_year,
            num_train_years=5,
            num_val_years=1,
            num_test_years=1,
            df_prices=df_prices,
            df_ret=df_ret,
            vol_df=df_vol,
        )
        # Backtest
        backtest_metrics, backtest_portfolio = backtest_agent(
            agent_path, test_data, drl_config
        )
        # Compute mean and std of validation rewards for this window
        rewards = window_rewards.get(window_idx + 1, [])
        val_reward_mean = float(pd.Series(rewards).mean()) if rewards else None
        val_reward_std = float(pd.Series(rewards).std()) if rewards else None
        all_backtest_results.append(
            {
                "window": window_idx + 1,
                "best_agent_path": os.path.basename(agent_path).split(".")[0],
                **backtest_metrics,
                "val_reward_mean": val_reward_mean,
                "val_reward_std": val_reward_std,
            }
        )
        all_portfolios[window_idx] = backtest_portfolio
        fname = f"portfolio_{window_idx + 1}_test={current_start_year + 6}.csv"
        backtest_portfolio.get_history().to_csv(
            os.path.join(drl_config.model_save_dir, fname)
        )

    results_filename = "backtest_results_summary_eval.csv"
    results_save_path = os.path.join(drl_config.model_save_dir, results_filename)
    results_df = pd.DataFrame(all_backtest_results)
    results_df.to_csv(results_save_path, index=False)

    return results_df, all_portfolios
