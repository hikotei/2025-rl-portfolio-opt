import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize

from sklearn.covariance import LedoitWolf
from pypfopt.efficient_frontier import EfficientFrontier
import pypfopt.objective_functions as objective_functions


class MVOOptimizer:
    """
    Mean-Variance Optimizer using PyPortfolioOpt and Ledoit-Wolf shrinkage.
    Optimizes portfolio weights by maximizing the Sharpe ratio.
    """

    def __init__(self, tickers, lookback=60, risk_free_rate=0.0):
        """
        Initializes the mean-variance optimizer with asset tickers, lookback window, and risk-free rate.
        
        Args:
            tickers: List of asset tickers to include in the portfolio.
            lookback: Number of days of historical returns to use for optimization.
            risk_free_rate: Annualized risk-free rate used in Sharpe ratio calculations.
        """
        self.tickers = tickers
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = (1 + risk_free_rate) ** (
            1 / 252
        ) - 1  # Convert annual to daily
        self.reset()

    def reset(self):
        """
        Resets the optimizer's internal state, clearing cash, shares, portfolio value, and history.
        """
        self.cash = 0
        self.shares = {t: 0 for t in self.tickers}
        self.portfolio_value = 0
        self.history = []

    @staticmethod
    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        """
        Calculates the negative Sharpe ratio for a given portfolio allocation.
        
        The negative Sharpe ratio is used as an objective function for optimization, where the Sharpe ratio is defined as the excess expected return over the risk-free rate divided by the portfolio's standard deviation.
        
        Args:
            weights: Portfolio weights for each asset.
            mean_returns: Expected returns for each asset.
            cov_matrix: Covariance matrix of asset returns.
            risk_free_rate: Risk-free rate used in Sharpe ratio calculation.
        
        Returns:
            The negative Sharpe ratio for the given portfolio weights.
        """
        weights = np.array(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return -sharpe

    def get_weights(self, return_window, method="pypfopt"):
        """
        Calculates optimal portfolio weights for a given window of historical returns using mean-variance optimization.
        
        If any returns are NaN, returns None. Uses Ledoit-Wolf shrinkage to estimate the covariance matrix and ensures it is positive semi-definite. If all mean returns are negative, assigns zero weights to all assets.
        
        Supports two optimization methods:
        - "pypfopt": Uses PyPortfolioOpt's EfficientFrontier to maximize the Sharpe ratio.
        - "scipy": Uses SciPy's SLSQP optimizer to minimize the negative Sharpe ratio, subject to weights summing to one and being between 0 and 1.
        
        Args:
            return_window (pd.DataFrame): Historical returns for the lookback period (rows: days, columns: assets).
            method (str): Optimization method, either "pypfopt" or "scipy".
        
        Returns:
            dict: Mapping of asset tickers to portfolio weights, or None if input contains NaNs or optimization fails.
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

    def backtest(self, df_ret, df_prices, start_date, end_date, initial_cash=100_000, method="pypfopt"):
        """
        Runs a backtest simulating portfolio rebalancing using mean-variance optimization over a specified date range.
        
        At each business day, computes optimal portfolio weights based on a lookback window of historical returns, reallocates capital according to these weights using current prices, and tracks portfolio value, cash, and asset holdings over time.
        
        Args:
            df_ret (pd.DataFrame): Daily returns for each asset.
            df_prices (pd.DataFrame): Daily prices for each asset.
            start_date (str or pd.Timestamp): Start date for the backtest.
            end_date (str or pd.Timestamp): End date for the backtest.
            initial_cash (float): Initial portfolio cash value.
            method (str): Optimization method to use ("pypfopt" or "scipy").
        
        Returns:
            pd.DataFrame: Time series of portfolio history, including value, cash, weights, and shares held for each asset.
        """
        # Validate date range
        if start_date > end_date:
            print("Warning: Start date is greater than end date")
            return pd.DataFrame()

        date_range = pd.bdate_range(start=start_date, end=end_date)
        self.reset()
        self.cash = initial_cash
        self.portfolio_value = initial_cash

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

            # Compute current portfolio value
            self.portfolio_value = (
                sum(self.shares.get(t, 0) * prices_today[t] for t in valid_tickers)
                + self.cash
            )
            # self.shares.get(t, 0) adresses the problem that when we have a new ticker,
            # eg XLC is introduced in 2019, we need to initialize shares[XLC] to 0

            # Get new weights
            weights = self.get_weights(return_window)
            # Reallocate capital using computed weights
            asset_cash = {t: weights[t] * self.portfolio_value for t in valid_tickers}
            # Rebalance shares to ensure integer values
            self.shares = {
                t: np.floor(asset_cash[t] / prices_today[t])
                if prices_today[t] != 0
                else 0
                for t in valid_tickers
            }
            # Allocate the rest of portfolio value to cash
            invested = sum(
                self.shares.get(t, 0) * prices_today[t] for t in valid_tickers
            )
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
