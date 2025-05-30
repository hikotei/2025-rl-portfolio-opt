from copy import deepcopy
from datetime import datetime
import os
from utils.drl_agent import DRLAgent

def generate_sliding_windows(start_year=2006, end_year=2021):
    """
    Yields (train, validation, test) date ranges.
    Each window: 5 years train + 1 year validation + 1 year test.
    """
    for i in range(end_year - start_year - 6 + 1):
        train_start = f"{start_year + i}-01-01"
        train_end = f"{start_year + i + 5}-01-01"
        val_start = train_end
        val_end = f"{start_year + i + 6}-01-01"
        test_start = val_end
        test_end = f"{start_year + i + 7}-01-01"
        yield dict(train=(train_start, train_end),
                   val=(val_start, val_end),
                   test=(test_start, test_end))

def make_env(PortfolioEnv, returns, prices, vol):
    """
    Constructs a PortfolioEnv instance from data slices.
    """
    return PortfolioEnv(
        returns_df=returns,
        prices_df=prices,
        vol_df=vol,
        window_size=60,
        transaction_cost=0,
        initial_balance=100_000,
        reward_scaling=1.0,
        eta=1 / 252,
    )

def train_agents(DRLAgent, PortfolioEnv, returns_df, prices_df, vol_df, window, n_seeds=5, total_timesteps=7_500_000):
    """
    Trains multiple agents with different seeds and returns them all.
    """
    agents = []

    train_returns = returns_df[window["train"][0]:window["train"][1]]
    train_prices = prices_df[window["train"][0]:window["train"][1]]
    train_vol = vol_df[window["train"][0]:window["train"][1]]

    for seed in range(n_seeds):
        train_env = make_env(PortfolioEnv, train_returns, train_prices, train_vol)

        agent = DRLAgent(
            env=train_env,
            model_name='ppo',
            n_envs=10,
            n_steps=756,
            batch_size=1260,
            n_epochs=16,
            learning_rate=3e-4,
            gamma=0.9,
            gae_lambda=0.9,
            clip_range=0.25,
        )
        agent.train(total_timesteps=total_timesteps, seed=seed)
        agents.append(agent)

    return agents

def select_best_agent(agents, PortfolioEnv, returns_df, prices_df, vol_df, window):
    """
    Selects the best agent based on Sharpe ratio evaluated on the validation period.
    """
    val_returns = returns_df[window["val"][0]:window["val"][1]]
    val_prices = prices_df[window["val"][0]:window["val"][1]]
    val_vol = vol_df[window["val"][0]:window["val"][1]]
    val_env = make_env(PortfolioEnv, val_returns, val_prices, val_vol)

    best_agent = None
    best_score = -float("inf")
    best_model_path = None

    for i, agent in enumerate(agents):
        metrics = agent.evaluate(val_env, n_episodes=1)
        val_score = metrics.get("Sharpe ratio", -float("inf"))

        if val_score > best_score:
            # Save model weights to temporary file
            temp_path = f"temp_model_{i}.zip"
            agent.model.save(temp_path)
            best_model_path = temp_path
            best_score = val_score

    # Create new agent with best model
    if best_model_path is not None:
        train_returns = returns_df[window["train"][0]:window["train"][1]]
        train_prices = prices_df[window["train"][0]:window["train"][1]]
        train_vol = vol_df[window["train"][0]:window["train"][1]]
        train_env = make_env(PortfolioEnv, train_returns, train_prices, train_vol)
        
        best_agent = DRLAgent(
            env=train_env,
            model_name='ppo',
            n_envs=10,
            n_steps=756,
            batch_size=1260,
            n_epochs=16,
            learning_rate=3e-4,
            gamma=0.9,
            gae_lambda=0.9,
            clip_range=0.25,
        )
        best_agent.model = best_agent.model.load(best_model_path)
        
        # Clean up temporary file
        os.remove(best_model_path)

    return best_agent

def backtest_all_agents(agents, PortfolioEnv, returns_df, prices_df, vol_df, window):
    """
    Evaluates all agents on the test period and returns their metrics.
    """
    test_returns = returns_df[window["test"][0]:window["test"][1]]
    test_prices = prices_df[window["test"][0]:window["test"][1]]
    test_vol = vol_df[window["test"][0]:window["test"][1]]

    test_env = make_env(PortfolioEnv, test_returns, test_prices, test_vol)
    print(f"Backtesting all agents on {window['test'][0]} to {window['test'][1]}")
    all_metrics = []

    for agent in agents:
        metrics = agent.evaluate(test_env, n_episodes=1)
        all_metrics.append(metrics)

    return all_metrics