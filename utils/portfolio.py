import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, tickers, initial_balance):
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_balance = self.initial_balance
        self.positions = {asset: 0.0 for asset in self.tickers}
        self.weights = {asset: 0.0 for asset in self.tickers}
        self.cash = self.initial_balance
        self.w_c = 1.0
        self.history = []
        self.purchases = {asset: 0.0 for asset in self.tickers}
        self.sales = {asset: 0.0 for asset in self.tickers}
        self.last_rebalance_date = None

    def __str__(self):
        return f"Portfolio with {len(self.tickers)} assets, initial value: ${self.initial_balance:,.2f}, current value: ${self.current_balance:,.2f}"

    def __repr__(self):
        return self.__str__()

    def print_summary(self):
        """
        Print out the portfolio's configuration and current settings.
        """
        print("\n=== Current State ===")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Portfolio Value: ${self.current_balance:,.2f}")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Cash Weight: {self.w_c:.2%}")

        data = []
        for ticker, shares in self.positions.items():
            if shares > 0:
                weight = self.weights[ticker]
                value = weight * self.current_balance
                data.append(
                    {
                        "Asset": ticker,
                        "Shares": shares,
                        "Weight": f"{weight:.2%}",
                        "Value": f"$ {value:,.2f}",
                    }
                )
        df = pd.DataFrame(data)
        df = df[["Asset", "Shares", "Weight", "Value"]]

        print("\nCurrent Holdings:")
        print(df.to_markdown(index=False))

        if self.history:
            print("\n=== Performance Metrics ===")
            metrics = self.calc_metrics()
            df_metrics = pd.DataFrame([metrics]).T
            df_metrics.index.name = "Metrics"
            df_metrics.columns = ["Value"]
            print(df_metrics.to_markdown())

        print("\n=== History & Configuration ===")
        print(f"Number of Records: {len(self.history)}")
        if self.history:
            first_date = self.history[0]["date"]
            last_date = self.history[-1]["date"]
            print(f"Date Range: {first_date} to {last_date}")

        print("\nAssets:")
        for ticker in self.tickers:
            print(f"  - {ticker}")

    def update(self, current_prices, date=None):
        """
        Update portfolio value and weights based on new prices without rebalancing.

        Args:
            current_prices (pd.Series): Current prices for each asset
            date: Optional date to record state (if None, state won't be recorded)
        """
        valid_tickers = set(self.tickers) & set(current_prices.index)

        # Calculate total portfolio value
        self.current_balance = (
            sum(self.positions.get(t, 0) * current_prices[t] for t in valid_tickers)
            + self.cash
        )

        # Update cash weight
        self.w_c = self.cash / self.current_balance if self.current_balance > 0 else 1.0

        # Update actual weights based on current prices
        if self.current_balance > 0:
            for t in valid_tickers:
                asset_value = self.positions[t] * current_prices[t]
                self.weights[t] = asset_value / self.current_balance
            # Set weights to 0 for invalid tickers
            for t in set(self.tickers) - valid_tickers:
                self.weights[t] = 0.0
        else:
            self.weights = {t: 0.0 for t in self.tickers}

        # Record state if date is provided
        if date is not None:
            record = {
                "date": date,
                "portfolio_value": self.current_balance,
                "cash": self.cash,
                "w_c": self.w_c,
            }

            for t in self.tickers:
                record[f"w_{t}"] = round(self.weights[t], 4)
            for t in self.tickers:
                record[f"s_{t}"] = round(self.positions.get(t, 0))

            self.history.append(record)

    def update_rebalance(self, current_prices, target_weights, date=None):
        """
        Rebalance portfolio according to target weights and record the state.

        Args:
            current_prices (pd.Series): Current prices for each asset
            target_weights (dict): Dictionary of {ticker: weight}
            date: Optional date to record state (if None, state won't be recorded)
        """
        valid_tickers = (
            set(self.tickers) & set(target_weights.keys()) & set(current_prices.index)
        )

        # Calculate total portfolio value
        self.current_balance = (
            sum(self.positions.get(t, 0) * current_prices[t] for t in valid_tickers)
            + self.cash
        )

        # Reallocate capital using computed weights
        target_asset_values = {
            t: target_weights[t] * self.current_balance for t in valid_tickers
        }

        # Calculate current shares and values
        current_shares = {t: self.positions.get(t, 0) for t in valid_tickers}

        # Calculate target shares (floored to integers)
        target_shares = {
            t: np.floor(target_asset_values[t] / current_prices[t])
            if current_prices[t] != 0
            else 0
            for t in valid_tickers
        }

        # Calculate purchases and sales based on actual share changes
        for t in valid_tickers:
            share_diff = target_shares[t] - current_shares[t]
            if share_diff > 0:
                # Purchase
                self.purchases[t] += share_diff * current_prices[t]
            elif share_diff < 0:
                # Sale
                self.sales[t] += abs(share_diff) * current_prices[t]

        # Update holdings with new shares
        self.positions = target_shares

        # Calculate actual invested amount after integer share conversion
        invested = sum(self.positions[t] * current_prices[t] for t in valid_tickers)
        self.cash = self.current_balance - invested

        # Update cash weight
        self.w_c = self.cash / self.current_balance if self.current_balance > 0 else 1.0

        # Update actual weights after integer share conversion
        if self.current_balance > 0:
            for t in valid_tickers:
                asset_value = self.positions[t] * current_prices[t]
                self.weights[t] = asset_value / self.current_balance
            # Set weights to 0 for invalid tickers
            for t in set(self.tickers) - valid_tickers:
                self.weights[t] = 0.0
        else:
            self.weights = {t: 0.0 for t in self.tickers}

        # Save updated portfolio state to history
        record = {
            "date": date,
            "portfolio_value": self.current_balance,
            "cash": self.cash,
            "w_c": self.w_c,
        }

        for t in self.tickers:
            record[f"w_{t}"] = round(self.weights[t], 4)
        for t in self.tickers:
            record[f"s_{t}"] = round(self.positions.get(t, 0))

        self.history.append(record)
        self.last_rebalance_date = date

    def get_return(self):
        """
        Get the most recent portfolio return.

        Returns:
            float: Portfolio return
        """
        if len(self.history) < 2:
            return 0.0

        prev_value = self.history[-2]["portfolio_value"]
        if prev_value <= 0:
            return 0.0

        return (self.current_balance - prev_value) / prev_value

    def get_history(self):
        """
        Get the portfolio history as a DataFrame.
        """
        return pd.DataFrame(self.history).set_index("date")

    def calc_metrics(self, risk_free_rate=0):
        """
        Calculate comprehensive portfolio performance metrics.

        Parameters:
        ----------
        risk_free_rate : float, optional
            The annualized risk-free rate used for calculating metrics like the
            Sharpe ratio and Sortino ratio.

        Returns:
        -------
        dict
            Dictionary containing various performance metrics including returns,
            risk metrics, and distribution statistics.
        """
        if not self.history:
            return {}

        df = pd.DataFrame(self.history).set_index("date")
        portfolio_values = df["portfolio_value"].values

        if len(portfolio_values) < 2:
            return {}

        # Calculate daily returns
        daily_returns = df["portfolio_value"].pct_change().dropna().values

        # Basic return metrics
        pv_initial = portfolio_values[0]
        pv_final = portfolio_values[-1]
        annual_return = (pv_final / pv_initial) ** (252 / len(daily_returns)) - 1
        cumulative_return = (pv_final / pv_initial) - 1

        # Volatility
        annual_volatility = np.std(daily_returns) * np.sqrt(252)

        # Risk-adjusted return metrics
        daily_risk_free_rate = risk_free_rate / 252
        excess_returns = daily_returns - daily_risk_free_rate
        std_dev_returns = np.std(daily_returns)

        sharpe_ratio = (
            np.mean(excess_returns) / std_dev_returns * np.sqrt(252)
            if std_dev_returns > 1e-9
            else np.nan
        )

        # Drawdown metrics
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = (
            np.min(drawdowns) if len(drawdowns) > 0 and np.any(rolling_max > 0) else 0.0
        )

        calmar_ratio = (
            annual_return / abs(max_drawdown)
            if abs(max_drawdown) > 1e-9 and pd.notna(annual_return)
            else np.nan
        )

        # Sortino ratio
        negative_returns = daily_returns[daily_returns < daily_risk_free_rate]
        if len(negative_returns) > 0:
            downside_std_dev = np.std(negative_returns)
            sortino_ratio = (
                np.mean(excess_returns) / downside_std_dev * np.sqrt(252)
                if downside_std_dev > 1e-9
                else np.nan
            )
        else:
            mean_er = np.mean(excess_returns)
            sortino_ratio = (
                np.inf if mean_er > 1e-9 else (0 if abs(mean_er) < 1e-9 else np.nan)
            )

        # Omega ratio
        threshold = daily_risk_free_rate
        gains = daily_returns[daily_returns > threshold] - threshold
        losses = daily_returns[daily_returns <= threshold] - threshold
        sum_gains = np.sum(gains)
        sum_abs_losses = abs(np.sum(losses))
        omega_ratio = (
            np.inf
            if sum_abs_losses < 1e-9 and sum_gains > 1e-9
            else (
                1
                if sum_abs_losses < 1e-9 and abs(sum_gains) < 1e-9
                else (sum_gains / sum_abs_losses if sum_abs_losses >= 1e-9 else np.nan)
            )
        )

        # Distribution statistics
        skew = pd.Series(daily_returns).skew()
        kurtosis = pd.Series(daily_returns).kurtosis()

        # Tail metrics
        tail_ratio = np.nan
        var_95 = np.nan
        if len(daily_returns) >= 20:
            percentile_5 = np.percentile(daily_returns, 5)
            percentile_95 = np.percentile(daily_returns, 95)
            var_95 = percentile_5
            if abs(percentile_5) < 1e-9:
                tail_ratio = (
                    np.inf
                    if percentile_95 > 1e-9
                    else (1 if abs(percentile_95) < 1e-9 else np.nan)
                )
            else:
                tail_ratio = percentile_95 / abs(percentile_5)

        # Stability
        stability = (
            1 / (1 + annual_volatility) if pd.notna(annual_volatility) else np.nan
        )

        # Calculate portfolio turnover
        total_purchases = sum(self.purchases.values())
        total_sales = sum(self.sales.values())
        portfolio_values = [record["portfolio_value"] for record in self.history]
        avg_portfolio_value = sum(portfolio_values) / len(portfolio_values)
        portfolio_turnover = min(total_purchases, total_sales) / avg_portfolio_value

        return {
            "Annual return": annual_return,
            "Cumulative returns": cumulative_return,
            "Annual volatility": annual_volatility,
            "Sharpe ratio": sharpe_ratio,
            "Calmar ratio": calmar_ratio,
            "Stability": stability,
            "Max drawdown": max_drawdown,
            "Omega ratio": omega_ratio,
            "Sortino ratio": sortino_ratio,
            "Skew": skew,
            "Kurtosis": kurtosis,
            "Tail ratio": tail_ratio,
            "Daily value at risk (95%)": var_95,
            "Portfolio turnover (in %)": portfolio_turnover,
        }
