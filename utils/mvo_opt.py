import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from pypfopt.efficient_frontier import EfficientFrontier
import pypfopt.objective_functions as objective_functions


class MVOOptimizer:
    """
    Mean-Variance Optimizer using PyPortfolioOpt and Ledoit-Wolf shrinkage.
    Optimizes portfolio weights by maximizing the Sharpe ratio.
    """

    def __init__(self, tickers, lookback=60):
        """
        Args:
            tickers (list): List of asset tickers.
            lookback (int): Number of days to use for historical estimation.
        """
        self.tickers = tickers
        self.lookback = lookback
        self.reset()

    def reset(self):
        """Reset the optimizer state."""
        self.cash = 0
        self.shares = {t: 0 for t in self.tickers}
        self.portfolio_value = 0
        self.history = []

    def get_weights(self, return_window):
        """
        Computes the optimal portfolio weights for the given return window.

        Args:
            return_window (pd.DataFrame): Lookback window of returns [days x assets].

        Returns:
            dict: Dictionary of {ticker: weight} or None if optimization fails.
        """
        if return_window.isnull().any().any():
            print(f"Warning: NaN values found in return window from {return_window.index.min()} to {return_window.index.max()}")
            return None

        try:
            lw = LedoitWolf()
            lw.fit(return_window)
            cov_matrix = lw.covariance_

            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals[eigvals < 0] = 0
            cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

            mu = return_window.mean()
            if (mu < 0).all():
                # print(f"All returns are negative for {return_window.index.min()} to {return_window.index.max()}")
                return {t: 0 for t in return_window.columns}

            ef = EfficientFrontier(mu, cov_psd)
            weights_array = ef.nonconvex_objective(
                objective_functions.sharpe_ratio,
                objective_args=(ef.expected_returns, ef.cov_matrix),
                weights_sum_to_one=True,
            )

            return dict(weights_array)
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            return None

    def backtest(self, df_ret, df_prices, start_date, end_date, initial_cash=100_000):
        """
        Simulates portfolio rebalancing over time using mean-variance optimization.

        - start with 100_000 cash
        - take lookback window of returns to compute weights using MVO
        - lookback window = today and lookback days before
        - use prices of today to rebalance weights to ensure integer values for shares
        - at the start of the next day, update portfolio value using new prices
        - which reflects the change in prices of the assets

        Args:
            df_ret (pd.DataFrame): Daily returns of assets.
            df_prices (pd.DataFrame): Daily prices of assets.
            start_date (str or pd.Timestamp): Backtest start date.
            end_date (str or pd.Timestamp): Backtest end date.
            initial_cash (float): Starting capital.

        Returns:
            pd.DataFrame: Portfolio history including weights, shares, and cash allocations.
        """
        # Validate date range
        if start_date > end_date:
            print("Warning: Start date is greater than end date")
            return pd.DataFrame()

        date_range = pd.bdate_range(start=start_date, end=end_date)
        self.reset()
        self.cash = initial_cash
        self.portfolio_value = initial_cash

        for eval_date in date_range:
            if eval_date not in df_ret.index or eval_date not in df_prices.index:
                # print(f"Date {eval_date} not in returns or prices")
                continue

            ret_idx = df_ret.index.get_loc(eval_date)
            if ret_idx < self.lookback:
                print(f"Not enough data for {eval_date}")
                continue

            return_window = df_ret.iloc[ret_idx - self.lookback : ret_idx]
            prices_today = df_prices.loc[eval_date].fillna(0)

            # Drop NaN tickers
            valid_tickers = [t for t in self.tickers if return_window[t].notna().all()]
            return_window = return_window[valid_tickers]
            prices_today = prices_today[valid_tickers]

            # Compute current portfolio value
            self.portfolio_value = (
                sum(self.shares.get(t, 0) * prices_today[t] for t in valid_tickers) + self.cash
            )
            # self.shares.get(t, 0) adresses the problem that when we have a new ticker,
            # eg XLC is introduced in 2019, we need to initialize shares[XLC] to 0

            # Get new weights
            weights = self.get_weights(return_window)
            # Reallocate capital using computed weights
            asset_cash = {t: weights[t] * self.portfolio_value for t in valid_tickers}
            # Rebalance shares to ensure integer values
            self.shares = {
                t: np.floor(asset_cash[t] / prices_today[t]) if prices_today[t] != 0 else 0
                for t in valid_tickers
            }
            # Allocate the rest of portfolio value to cash
            invested = sum(self.shares.get(t, 0) * prices_today[t] for t in valid_tickers)
            self.cash = self.portfolio_value - invested

            # Save history
            record = {
                "date": eval_date,
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "lookback": self.lookback,
                "w_c": self.cash / self.portfolio_value,
            }
            for t in valid_tickers:
                record[f"w_{t}"] = weights[t]
            for t in self.tickers:
                record[f"shares_{t}"] = self.shares.get(t, 0)

            self.history.append(record)

        return pd.DataFrame(self.history).set_index("date")
