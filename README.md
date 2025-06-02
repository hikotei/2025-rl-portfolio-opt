# 📈 Portfolio Optimization Framework

Implements two portfolio allocation strategies:

- **Mean-Variance Optimization (MVO)** via [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
- **Deep Reinforcement Learning (DRL)** using PPO from Stable-Baselines3

Based on [Sood et al. (2023)](https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf): *"Deep Reinforcement Learning for Optimal Portfolio Allocation."*

## 🧠 Methods Overview

Portfolio optimization is a core task in financial management — it involves dynamically allocating capital across a set of assets to achieve a balance between return maximization and risk minimization.

### 🧮 Mean-Variance Optimization (MVO)

- 60-day rolling window of returns
- Covariance estimated using **Ledoit-Wolf shrinkage**
- Optimizes **Sharpe Ratio** with long-only constraints
- Rebalanced **daily**

**Limitations**:
- Sensitive to estimation error
- Assumes static linear relationships
- Not adaptive to market regimes

### 🚀 Deep Reinforcement Learning (DRL)

- Frames portfolio allocation as a sequential decision problem
- Trains PPO agent on historical data with simulated market replay
- Reward: **Differential Sharpe Ratio** (Moody et al., 1998)

**Advantages**:
- Learns non-linear, dynamic relationships
- Directly optimizes risk-adjusted returns
- Adapts to non-stationary environments

## 📁 Project Structure

```
portfolio_opt/
├── data/                       # Market data files
├── notebooks/                  # Jupyter notebooks for analysis
├── utils/                      # Core implementation
│   ├── drl_agent.py           # DRL Agent (PPO) wrapper
│   ├── mvo_strategy.py        # MVO strategy implementation
│   ├── portfolio.py           # Portfolio management
│   └── portfolio_env.py       # Custom Gym environment
├── models/                     # Trained DRL models
├── logs/                      # Training logs
└── results/                   # Backtest results and plots
```

## ⚙️ Core Components

### Portfolio Management
`Portfolio` (utils/portfolio.py)
- Manages portfolio state and rebalancing
- Tracks positions, weights, and cash
- Calculates performance metrics
- Handles integer share constraints

### Mean-Variance Optimization
`MVOPortfolio` (utils/mvo_strategy.py)
- Implements rolling-window MVO strategy
- Uses Ledoit-Wolf shrinkage for robust covariance estimation
- Optimizes Sharpe ratio with long-only constraints
- Supports both PyPortfolioOpt and SciPy optimization methods
- Daily rebalancing with integer share constraints

**Training / Setup**
*   Rolling optimization with 60-day lookback.
*   Daily rebalancing.
*   Uses Ledoit-Wolf shrinkage estimator for covariances (to avoid noisy estimates).
*   Optimization objective: Maximize Sharpe Ratio.
*   Solved via PyPortfolioOpt.


### Deep Reinforcement Learning
`DRLAgent` (utils/drl_agent.py)
- Wraps PPO implementation from Stable-Baselines3
- Supports parallel training with SubprocVecEnv
- Configurable hyperparameters and policy architecture
- Handles model saving/loading and evaluation

`PortfolioEnv` (utils/portfolio_env.py)
- Custom Gym environment for portfolio management
- State space: 60-day lookback window of log returns
- Additional features: volatility indicators (vol20, VIX, vol_ratio)
- Reward: Differential Sharpe Ratio
- Action space: Portfolio weights (continuous)

**DRL Training**
- 10 sliding windows (7 years each)
  - 5 years training
  - 1 year validation
  - 1 year backtest
- 5 seeds per window (50 total agents)
- 7.5M timesteps per training round

**PPO Hyperparameters**
- Policy: [64, 64] MLP with Tanh activation
- Learning rate: Linear decay (3e-4 to 1e-5)
- Batch size: 1260 (252 × 5)
- Gamma: 0.9
- GAE lambda: 0.9
- Clip range: 0.25