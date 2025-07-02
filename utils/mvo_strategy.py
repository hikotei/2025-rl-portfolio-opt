import numpy as np
import pandas as pd
from tqdm import tqdm

from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf as SklearnLedoitWolf

from skfolio import RiskMeasure
from skfolio.prior import EmpiricalPrior
from skfolio.moments import EmpiricalMu, LedoitWolf as SkfolioLedoitWolf
from skfolio.optimization import MeanRisk, ObjectiveFunction

from utils.portfolio import Portfolio


class MVOStrategy:
    """
    Mean-Variance Optimization (MVO) based portfolio strategy.
    Implements a portfolio management strategy that uses MVO to optimize asset weights
    by maximizing the Sharpe ratio, with Ledoit-Wolf shrinkage for covariance estimation.
    """

    def __init__(self, tickers, lookback=60, risk_free_rate=0.0, initial_cash=100_000, freq='daily'):
        """
        Args:
            tickers (list): List of asset tickers.
            lookback (int): Number of periods to use for historical estimation.
            risk_free_rate (float): Annual risk-free rate (default: 0.0).
            initial_cash (float): Initial cash amount for the portfolio (default: 100_000).
            freq (str): Data frequency, one of ['daily', 'monthly', 'quarterly', 'yearly']. Default is 'daily'.
        """
        self.tickers = tickers
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        
        # Set annualization factor based on frequency
        freq_factors = {
            'daily': 252,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        if freq not in freq_factors:
            raise ValueError(f"freq must be one of {list(freq_factors.keys())}")
        self.annualization_factor = freq_factors[freq]
        
        # Convert annual risk-free rate to per-period rate
        self.periodic_risk_free = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1
        
        self.portfolio = Portfolio(tickers, initial_cash)
        self.skipped_dates = []

    @staticmethod
    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        """
        Calculate the negative Sharpe ratio (to be minimized).
        Sharpe Ratio = (Expected Return - Risk Free Rate) / Portfolio Standard Deviation
        
        The risk-free rate used here is already converted to the appropriate frequency (daily/monthly/etc)
        based on the annualization factor.
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

        mu = return_window.mean()
        if (mu < 0).all():
            self.skipped_dates.extend(return_window.index)
            return None

        if method == "skfolio":
            model = MeanRisk(
                risk_measure=RiskMeasure.STANDARD_DEVIATION,
                objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                prior_estimator=EmpiricalPrior(
                    mu_estimator=EmpiricalMu(
                        window_size=None
                    ),  # by default uses all given data
                    covariance_estimator=SkfolioLedoitWolf(),
                ),
                portfolio_params=dict(name="Max Sharpe"),
                # solver_params=dict(verbose=True)
            )
            model.fit(return_window)
            weights = model.weights_
            weights_dict = dict(zip(return_window.columns, weights))
            return weights_dict

        else:
            lw = SklearnLedoitWolf()
            lw.fit(return_window)
            cov_matrix = lw.covariance_

            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals[eigvals < 0] = 0
            cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # ensure symmetry
            cov_psd = (cov_psd + cov_psd.T) / 2

            if method == "pypfopt":
                ef = EfficientFrontier(mu, cov_psd)
                weights_array = ef.nonconvex_objective(
                    objective_functions.sharpe_ratio,
                    objective_args=(ef.expected_returns, ef.cov_matrix),
                    weights_sum_to_one=True,
                )
                weights_dict = dict(weights_array)
                return weights_dict

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
                    args=(mu, cov_matrix, self.periodic_risk_free),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )

                if not result["success"]:
                    print(f"Warning: Optimization failed: {result['message']}")
                    return {t: 0 for t in return_window.columns}

                weights_dict = dict(zip(return_window.columns, result["x"]))
                return weights_dict

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


class NaiveStrategy:
    """
    Naive 1/N portfolio strategy
    at each iteration, rebalance to 1/N weights based on current prices and portfolio value
    """

    def __init__(self, tickers, initial_cash=100_000):
        """
        Args:
            tickers (list): List of asset tickers.
            initial_cash (float): Initial cash amount for the portfolio.
        """
        self.tickers = tickers
        self.portfolio = Portfolio(tickers, initial_cash)
        self.skipped_dates = []

    def get_weights(self, return_window):
        """
        Computes equal weights (1/N) for all assets.
        """
        valid_tickers = [
            t for t in return_window.columns if return_window[t].notna().all()
        ]
        n_assets = len(valid_tickers)
        if n_assets == 0:
            return None
        
        equal_weight = 1.0 / n_assets
        return {ticker: equal_weight for ticker in valid_tickers}

    def backtest(
        self,
        df_ret,
        df_prices,
        start_date,
        end_date,
    ):
        """
        Simulates portfolio over time using naive 1/N strategy.

        Args:
            df_ret (pd.DataFrame): Daily returns of assets.
            df_prices (pd.DataFrame): Daily prices of assets.
            start_date (str or pd.Timestamp): Backtest start date.
            end_date (str or pd.Timestamp): Backtest end date.

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
                continue

            prices_today = df_prices.loc[eval_date].fillna(0)
            
            # Use a single day of returns to check for valid tickers
            return_window = df_ret.loc[[eval_date]]
            
            # Drop NaN tickers
            valid_tickers = [t for t in self.tickers if return_window[t].notna().all()]
            return_window = return_window[valid_tickers]
            prices_today = prices_today[valid_tickers]

            # Get equal weights
            weights = self.get_weights(return_window)
            if weights is None:
                # no new weights, but still update portfolio value with current prices
                # using update function
                self.portfolio.update(prices_today, date=eval_date)
                continue

            # Rebalance portfolio
            self.portfolio.update_rebalance(prices_today, weights, date=eval_date)

        return self.portfolio
