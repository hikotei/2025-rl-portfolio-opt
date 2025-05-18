import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.efficient_frontier import EfficientFrontier
import pypfopt.objective_functions as objective_functions

# ------------------------------
# Parameters and Tickers
# ------------------------------
sector_tickers = [
    "XLF",
    "XLK",
    "XLV",
    "XLY",
    "XLP",
    "XLE",
    "XLI",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
]

lookback = 60
initial_cash = 100_000
start_date = pd.to_datetime("2019-01-01")
end_date = pd.to_datetime("2020-01-01")
date_range = pd.bdate_range(start=start_date, end=end_date)

# ------------------------------
# Load Data
# ------------------------------
df_ret = pd.read_parquet("../data/returns.parquet")
df_prices = pd.read_parquet("../data/prices.parquet")
df_vol = pd.read_parquet("../data/vola.parquet")

# ------------------------------
# Initialize Portfolio
# ------------------------------
portfolio_value = initial_cash
portfolio_history = []
cash = initial_cash
shares = {t: 0 for t in sector_tickers}

# ------------------------------
# Main Backtest Loop
# ------------------------------
for eval_date in date_range:
    if eval_date not in df_ret.index:
        print(f"Date {eval_date} not in data, skipping.")
        continue

    ret_idx = df_ret.index.get_loc(eval_date)
    prices_idx = df_prices.index.get_loc(eval_date)

    if ret_idx < lookback:
        print(f"Not enough data for {eval_date}, skipping.")
        continue

    return_window = df_ret.iloc[ret_idx - lookback : ret_idx]
    prices_window = df_prices.iloc[prices_idx - lookback : prices_idx]

    nan_tickers = [t for t in sector_tickers if return_window[t].isna().any()]
    valid_sectors = [t for t in sector_tickers if t not in nan_tickers]

    if nan_tickers:
        print(f"NaN values for {eval_date}: {nan_tickers}")
        return_window = return_window.drop(columns=nan_tickers)
        prices_window = prices_window.drop(columns=nan_tickers)

    prices = df_prices.iloc[prices_idx].to_dict()
    if portfolio_history:
        portfolio_value = sum([shares[t] * prices[t] for t in valid_sectors]) + cash

    # ------------------------------
    # Estimate Covariance and Returns
    # ------------------------------
    lw = LedoitWolf()
    lw.fit(return_window)
    cov_matrix = lw.covariance_

    # Fix for numerical stability
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals[eigvals < 0] = 0
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    mu = return_window.mean()
    if (mu < 0).all():
        print(f"All expected returns are negative for {eval_date}")
        # Optional: continue

    # ------------------------------
    # Portfolio Optimization
    # ------------------------------
    ef = EfficientFrontier(mu, cov_psd)
    w_min, w_max = 0, 1
    weights_raw = ef.nonconvex_objective(
        objective_functions.sharpe_ratio,
        objective_args=(ef.expected_returns, ef.cov_matrix),
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: w - w_min},
            {"type": "ineq", "fun": lambda w: w_max - w},
        ],
    )

    # ------------------------------
    # Rebalancing and Recording
    # ------------------------------
    asset_cash = {t: weights_raw[t] * portfolio_value for t in valid_sectors}
    shares = {t: np.floor(asset_cash[t] / prices[t]) for t in valid_sectors}
    weights = {t: shares[t] * prices[t] / portfolio_value for t in valid_sectors}
    cash = portfolio_value - np.sum([shares[t] * prices[t] for t in valid_sectors])
    w_c = cash / portfolio_value

    portfolio_history.append(
        {"date": eval_date, "cash": cash, "portfolio_value": portfolio_value}
    )

# ------------------------------
# Final Output
# ------------------------------
portfolio_df = pd.DataFrame(portfolio_history)
print(portfolio_df)
