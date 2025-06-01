import numpy as np
import pandas as pd
from tqdm import tqdm

from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from utils.portfolio import Portfolio


class MVOPortfolio:
    """
    Mean-Variance Optimization (MVO) based portfolio strategy.
    Implements a portfolio management strategy that uses MVO to optimize asset weights
    by maximizing the Sharpe ratio, with Ledoit-Wolf shrinkage for covariance estimation.
    """

    def __init__(self, tickers, lookback=60, risk_free_rate=0.0, initial_cash=100_000):
        """
        Args:
            tickers (list): List of asset tickers.
            lookback (int): Number of days to use for historical estimation.
            risk_free_rate (float): Annual risk-free rate (default: 0.0).
            initial_cash (float): Initial cash amount for the portfolio (default: 100_000).
        """
        self.tickers = tickers
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
        self.portfolio = Portfolio(tickers, initial_cash)

    @staticmethod
    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        """
        Calculate the negative Sharpe ratio (to be minimized).
        Sharpe Ratio = (Expected Return - Risk Free Rate) / Portfolio Standard Deviation
        """
        weights = np.array(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return -sharpe

    def get_weights(self, return_window, method="pypfopt"):
        """
        Computes the optimal portfolio weights for the given return window.

        Args:
            return_window (pd.DataFrame): Lookback window of returns [days x assets].

        Returns:
            dict: Dictionary of {ticker: weight} or None if optimization fails.
        """
        if return_window.isnull().any().any():
            print(
                f"Warning: NaN values found in return window from {return_window.index.min()} to {return_window.index.max()}"
            )
            return None

        lw = LedoitWolf()
        lw.fit(return_window)
        cov_matrix = lw.covariance_

        # Ensure PSD
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals[eigvals < 0] = 0
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # ensure symmetry
        cov_psd = (cov_psd + cov_psd.T) / 2

        mu = return_window.mean()
        if (mu < 0).all():
            # print(f"All returns are negative for {return_window.index.min()} to {return_window.index.max()}")
            return {t: 0 for t in return_window.columns}

        if method == "pypfopt":
            ef = EfficientFrontier(mu, cov_psd)
            weights_array = ef.nonconvex_objective(
                objective_functions.sharpe_ratio,
                objective_args=(ef.expected_returns, ef.cov_matrix),
                weights_sum_to_one=True,
            )
            return dict(weights_array)

        if method == "scipy":
            n_assets = len(return_window.columns)

            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets

            # Constraints
            constraints = {
                "type": "eq",
                "fun": lambda x: np.sum(x) - 1,
            }  # weights sum to 1
            bounds = tuple((0, 1) for _ in range(n_assets))  # 0 <= weight <= 1

            # Optimize
            result = minimize(
                self.negative_sharpe_ratio,
                initial_weights,
                args=(mu, cov_matrix, self.daily_risk_free),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result["success"]:
                print(f"Warning: Optimization failed: {result['message']}")
                return {t: 0 for t in return_window.columns}

            return dict(zip(return_window.columns, result["x"]))

    def backtest(
        self,
        df_ret,
        df_prices,
        start_date,
        end_date,
        method="pypfopt",
    ):
        """
        Simulates portfolio over time using mean-variance optimization.

        Args:
            df_ret (pd.DataFrame): Daily returns of assets.
            df_prices (pd.DataFrame): Daily prices of assets.
            start_date (str or pd.Timestamp): Backtest start date.
            end_date (str or pd.Timestamp): Backtest end date.
            method (str): Optimization method to use ("pypfopt" or "scipy").

        Returns:
            pd.DataFrame: Portfolio history including weights, shares, and cash allocations.
        """
        # Validate date range
        if start_date > end_date:
            print("Warning: Start date is greater than end date")
            return pd.DataFrame()

        date_range = pd.bdate_range(start=start_date, end=end_date)
        self.portfolio.reset()

        for eval_date in tqdm(date_range, desc="Running backtest"):
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

            # Get new weights
            weights = self.get_weights(return_window, method=method)
            if weights is None:
                continue

            # Rebalance portfolio
            self.portfolio.update_rebalance(prices_today, weights, date=eval_date)

        return self.portfolio
