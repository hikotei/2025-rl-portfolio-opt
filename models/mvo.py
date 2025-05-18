import numpy as np
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

    def get_weights(self, return_window):
        """
        Computes the optimal portfolio weights for the given return window.

        Args:
            return_window (pd.DataFrame): Lookback window of returns [days x assets].

        Returns:
            dict: Dictionary of {ticker: weight} or None if optimization fails.
        """
        if return_window.isnull().any().any():
            return None

        tickers = return_window.columns.tolist()

        # Estimate covariance matrix with Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        lw.fit(return_window)
        cov_matrix = lw.covariance_

        # Ensure the covariance matrix is positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals[eigvals < 0] = 0  # truncate negative eigenvalues
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Estimate expected returns (simple average)
        mu = return_window.mean()

        # Edge case: all negative expected returns
        if (mu < 0).all():
            return None

        # Optimize for maximum Sharpe ratio
        ef = EfficientFrontier(mu, cov_psd)
        weights_array = ef.nonconvex_objective(
            objective_functions.sharpe_ratio,
            objective_args=(ef.expected_returns, ef.cov_matrix),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # sum(w) = 1
                {"type": "ineq", "fun": lambda w: w},  # w >= 0
                {"type": "ineq", "fun": lambda w: 1 - w},  # w <= 1
            ],
        )

        # TODO : do i need to convert to dict or is ordered dict also fine ?
        return dict(weights_array)
