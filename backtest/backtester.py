import pandas as pd
import numpy as np
from models.mvo import MVOOptimizer


class MVOBacktester:
    """
    Backtester for Mean-Variance Optimized portfolios.
    Rebalances daily based on a fixed lookback window and tracks portfolio evolution.
    """

    def __init__(self, returns, prices, tickers, lookback=60, initial_cash=100_000):
        """
        Args:
            returns (pd.DataFrame): Log returns [date x asset].
            prices (pd.DataFrame): Prices [date x asset].
            tickers (list): List of asset tickers.
            lookback (int): Lookback window in trading days.
            initial_cash (float): Starting cash amount.
        """
        self.returns = returns
        self.prices = prices
        self.tickers = tickers
        self.lookback = lookback
        self.initial_cash = initial_cash

        self.optimizer = MVOOptimizer(tickers, lookback)
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = {t: 0 for t in self.tickers}
        self.portfolio_value = self.initial_cash
        self.history = []

    def run(self, date_range):
        """
        Executes the backtest over the specified business date range.

        Args:
            date_range (pd.DatetimeIndex): Business days for evaluation.

        Returns:
            pd.DataFrame: Daily portfolio value and cash history.
        """
        for date in date_range:
            if date not in self.returns.index or date not in self.prices.index:
                continue

            idx = self.returns.index.get_loc(date)
            if idx < self.lookback:
                continue

            ret_window = self.returns.iloc[idx - self.lookback:idx]
            price_row = self.prices.loc[date]

            weights = self.optimizer.get_weights(ret_window)

            if weights is None:
                continue

            # Compute current portfolio value
            self.portfolio_value = sum(
                self.shares[t] * price_row[t] for t in self.tickers
            ) + self.cash

            # Reallocate capital using computed weights
            asset_cash = {t: weights[t] * self.portfolio_value for t in self.tickers}
            self.shares = {
                t: np.floor(asset_cash[t] / price_row[t]) for t in self.tickers
            }
            invested = sum(self.shares[t] * price_row[t] for t in self.tickers)
            self.cash = self.portfolio_value - invested
            self.portfolio_value = invested + self.cash

            # Save history
            self.history.append(
                {
                    "date": date,
                    "portfolio_value": self.portfolio_value,
                    "cash": self.cash,
                    **{f"{t}_shares": self.shares[t] for t in self.tickers},
                }
            )

        return pd.DataFrame(self.history).set_index("date")