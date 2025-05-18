# %%
import pandas as pd
from utils.portfolio_env_old import PortfolioEnv
from utils.drl_agent import DRLAgent

%load_ext autoreload
%autoreload 2

# %%
# --- Load data (replace with actual paths) ---
returns_df = pd.read_parquet("../data/returns.parquet")
prices_df = pd.read_parquet("../data/prices.parquet")
vol_df = pd.read_parquet("../data/vola.parquet")

# %%
# subset data to just one year 2020 - 2021
train_start = "2020-01-01"
train_end = "2020-06-01"

train_ret = returns_df[train_start:train_end]
train_prices = prices_df[train_start:train_end]
train_vol = vol_df[train_start:train_end]

# %%
# --- Create environment ---
env = PortfolioEnv(
    returns_df=train_ret,
    prices_df=train_prices,
    vol_df=train_vol,
    window_size=60,
    transaction_cost=0,
    initial_balance=100_000,
    reward_scaling=1.0,
    eta=1 / 252,
)

# %%
# = = = = = = = = 
# CHECK ENVIRONMENT
# = = = = = = = = 

# from stable_baselines3.common.env_checker import check_env
# check_env(env)

# UserWarning: Your observation  has an unconventional shape (neither an image, nor a 1D vector). 
# We recommend you to flatten the observation to have only a 1D vector or use a custom policy to properly process the data.

# UserWarning: We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) 
# cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

# = = = = = = = = 
# RANDOM AGENT
# = = = = = = = = 

# obs, info = env.reset()
# n_steps = 10
# for _ in range(n_steps):
#     action = env.action_space.sample() # random action
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(info)
#     if terminated:
#         obs, info = env.reset()

# %%
agent = DRLAgent(
    env,
    model_name='ppo',
    n_envs=5,
    n_steps=756,
    batch_size=1260,
    n_epochs=16,
    learning_rate=3e-4, # anneal to 1e-5
    gamma=0.9,
    gae_lambda=0.9,
    # clip_range=0.25
)

agent.train(total_timesteps=100)
agent.save("../models/ppo_portfolio.zip")

# %%
# subset data to just one year 2020 - 2021
eval_start = "2021-01-01"
eval_end = "2021-06-01"

eval_ret = returns_df[eval_start:eval_end]
eval_prices = prices_df[eval_start:eval_end]
eval_vol = vol_df[eval_start:eval_end]

# %%
# --- Evaluate DRL agent ---
# Create evaluation environment
eval_env = PortfolioEnv(
    returns_df=eval_ret,
    prices_df=eval_prices,
    vol_df=eval_vol,
    window_size=60,
    transaction_cost=0,
    initial_balance=100_000,
    reward_scaling=1.0,
    eta=1 / 252,
)

# Evaluate DRL agent
print("Evaluating DRL agent...")
drl_metrics = agent.evaluate(eval_env, n_episodes=1)


