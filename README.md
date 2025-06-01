# 📈 Portfolio Optimization Framework

This repository implements **portfolio optimization** using both:

- Mean-Variance Optimization (MVO) via PyPortfolioOpt
- Deep Reinforcement Learning (DRL) using PPO from Stable-Baselines3

Inspired by the 2023 paper by Sood et al: [*"Deep Reinforcement Learning for Optimal Portfolio Allocation"*.](https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf)

---

## 📁 Project Structure

```
portfolio_opt/
├── __init__.py
│
├── utils/
│   ├── mvo.py                  # MVOOptimizer class using Ledoit-Wolf + PyPortfolioOpt
│   └── drl_agent.py            # DRLAgent wrapper for PPO from Stable-Baselines3
│   └── portfolio_env.py        # Custom Gym-like environment for DRL training & evaluation
│
├── data/                       # Place for prices.csv, returns.csv, vola.csv
│
├── notebooks/
```

results folder is for plots etc
models saves the model zip files
logs for tensorboard log files

---

## ⚙️ Components

### 🧠 `MVOOptimizer` (models/mvo.py)
- Uses a 60-day rolling window of returns.
- Applies **Ledoit-Wolf shrinkage** for robust covariance estimation.
- Optimizes **Sharpe Ratio** with long-only constraints (0 ≤ w ≤ 1).
- Returns portfolio weights as a dictionary.

### 📊 `MVOBacktester` (backtest/backtester.py)
- Simulates realistic trading:
  - Rebalances daily
  - Allocates whole shares
  - Tracks cash and portfolio value
- Outputs a time-indexed DataFrame with portfolio history.

### 🧠 `DRLAgent` (models/drl_agent.py)
- Wraps PPO model from `Stable-Baselines3`.
- Supports training and inference on a custom `PortfolioEnv`.
- Uses differential Sharpe ratio as the reward function (like in the paper).
- Can handle multiple training seeds and sliding windows for backtesting.

### 🌍 `PortfolioEnv` (env/portfolio_env.py)
- Gym-compatible environment for portfolio management.
- Simulates market replay with rebalancing and portfolio tracking.
- Accepts action vectors (portfolio weights), returns reward and state.
- Includes volatility features (`vol20`, `VIX`, `vol_ratio`) and log-returns matrix.

---

## 🚀 Example Workflow

```bash
# Prepare data
...

# Run MVO backtest
...

# Train DRL agent
...
```

---

## 🧱 Next Steps

- Add transaction costs / slippage handling
- Add DRL strategy using PPO + Gym-compatible `PortfolioEnv`
- Integrate both into a common `Backtester` interface
- Add CLI and experiment config support (e.g. Hydra/YAML)

---

## 🧑‍💻 Author

Developed by [@hikotei](https://github.com/hikotei) with assistance from ChatGPT.
