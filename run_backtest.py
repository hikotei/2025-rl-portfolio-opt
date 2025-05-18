import pandas as pd
from backtest.backtester import MVOBacktester

# Load data
returns = pd.read_parquet("./data/returns.parquet")
prices = pd.read_parquet("./data/prices.parquet")
tickers = returns.columns.tolist()
date_range = pd.bdate_range("2019-01-01", "2020-01-01")

# Run backtest
backtester = MVOBacktester(returns, prices, tickers)
results_df = backtester.run(date_range)

# Inspect or plot
print(results_df.head())