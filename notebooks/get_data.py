import numpy as np
import pandas as pd
import yfinance as yf

# Sector ETF tickers representing the 11 S&P 500 sectors
sector_tickers = [
    "XLF",  # Financials
    "XLK",  # Technology
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLU",  # Utilities
    "XLB",  # Materials
    "XLRE",  # Real Estate
    "XLC",  # Communication Services
]

indices = [
    "^GSPC",  # S&P 500
    "^VIX",  # Volatility Index
]

# Date range from the paper
start_date = "2006-01-01"
end_date = "2021-12-31"

# Download daily adjusted close prices for sector ETFs
prices = yf.download(
    sector_tickers + indices,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=True,
    progress=True,
)["Close"]

# Compute log returns for sector prices and S&P 500
log_returns = np.log(prices / prices.shift(1))

# Drop the first row NaNs
log_returns = log_returns.dropna(axis=0, how="all")

# Calculate volatility metrics
sp500_returns = prices["^GSPC"].pct_change()  # simple returns
vol20 = sp500_returns.rolling(20).std()
vol60 = sp500_returns.rolling(60).std()
vol_ratio = vol20 / vol60

# Create df to hold vol metrics
vol_df = pd.DataFrame(
    {
        "vol20": vol20,
        "vol60": vol60,
        "vol_ratio": vol_ratio,
        "VIX": prices["^VIX"],
    }
)

# Drop first 60 rows to avoid NaN values
# vol_df = vol_df.dropna()

vol_df_std = vol_df.copy()
# Standardize the metrics using expanding lookback window to prevent look-ahead bias
for col in ['vol20', 'vol60', 'vol_ratio', 'VIX']:
    mean = vol_df[col].expanding().mean()
    std = vol_df[col].expanding().std()
    vol_df_std[col] = (vol_df[col] - mean) / std

# Drop the first row with NaN since there is no std yet
vol_df_std = vol_df_std.dropna(how='all')

# Save prices to CSV (contains all sectors and S&P 500)
prices[sector_tickers].to_csv('./data/prices.csv', index=True)

# Save returns to CSV (contains all sectors and S&P 500)
log_returns[sector_tickers].to_csv('./data/returns.csv', index=True)

# Save volatility indicators separately
vol_df_std[['vol20', 'vol_ratio', 'VIX']].to_csv('./data/vola.csv', index=True)