import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import diverging, sequential


def get_drl_portfolios(drl_model_dir: str):
    # get csv files in dir where fname contains portfolio
    csv_files = sorted(
        [
            f
            for f in os.listdir(drl_model_dir)
            if f.endswith(".csv") and "portfolio" in f
        ]
    )
    print(csv_files)

    # load csv files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(drl_model_dir, file))
        df.reset_index(inplace=True)
        # rename index to trading_day
        df.rename(columns={"index": "trading_day"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        dfs.append(df)

    df_drl_port = (
        pd.concat(dfs, ignore_index=True)
        .sort_values(by=["date"])
        .reset_index(drop=True)
    )

    return df_drl_port


def plot_portfolio_val(
    START_DATE: str,
    END_DATE: str,
    mvo_history_df: pd.DataFrame,
    naive_portfolio_df: pd.DataFrame = None,
    df_ref: pd.DataFrame = None,
    ref_ticker: str = None,
    initial_shares: float = None,
    save_dir: str = None,
    fname: str = None,
    title: str = None,
):
    # reset sns style
    sns.reset_defaults()
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # PLOT Portfolio Value Development
    plt.figure(figsize=(15, 5))
    palette = sns.color_palette(
        "coolwarm", as_cmap=False, n_colors=mvo_history_df["lookback"].nunique()
    )
    sns.lineplot(
        data=mvo_history_df,
        x="date",
        y="portfolio_value",
        hue="lookback",
        palette=palette,
        lw=1,
    )

    # plot naive strategy
    if naive_portfolio_df is not None:
        sns.lineplot(
            data=naive_portfolio_df,
            x="date",
            y="portfolio_value",
            c="purple",
            label="Naive",
            lw=1.5,
        )

    # plot direct investment in SPY
    plt.plot(
        df_ref[ref_ticker].loc[START_DATE:END_DATE] * initial_shares,
        c="k",
        label=ref_ticker,
        lw=1.5,
    )

    if title is not None:
        plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    # save plot
    plt.savefig(f"{save_dir}/{fname}.pdf", dpi=300)
    plt.show()


def plot_portfolio_val_interactive(
    START_DATE: str,
    END_DATE: str,
    mvo_history_df: pd.DataFrame,
    naive_portfolio_df: pd.DataFrame = None,
    df_ref: pd.DataFrame = None,
    ref_ticker: str = None,
    initial_shares: float = None,
):
    # Initialize figure
    fig = go.Figure()

    # Add portfolio lines by lookback
    unique_lookbacks = mvo_history_df["lookback"].unique()
    colors = px.colors.sample_colorscale(
        "RdBu", [i / (len(unique_lookbacks) - 1) for i in range(len(unique_lookbacks))]
    )

    for i, lookback in enumerate(unique_lookbacks):
        df_subset = mvo_history_df[mvo_history_df["lookback"] == lookback]
        fig.add_trace(
            go.Scatter(
                x=df_subset.index,
                y=df_subset["portfolio_value"],
                mode="lines",
                line=dict(color=colors[i], width=1.5),
                name=f"Lookback {lookback}",
                hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

    # Add naive strategy
    if naive_portfolio_df is not None:
        fig.add_trace(
            go.Scatter(
                x=naive_portfolio_df.index,
                y=naive_portfolio_df["portfolio_value"],
                mode="lines",
                line=dict(color="purple", width=1.5),
                name="Naive Strategy",
            )
        )

    # Add direct investment in SPY
    spy_series = df_ref[ref_ticker].loc[START_DATE:END_DATE] * initial_shares
    fig.add_trace(
        go.Scatter(
            x=spy_series.index,
            y=spy_series.values,
            mode="lines",
            line=dict(color="black", width=1.5),
            name=ref_ticker,
        )
    )

    # Update layout
    fig.update_layout(
        title="📈 Portfolio Value Development by Lookback Period",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        legend_title="Strategy",
        height=600,
        width=1200,
    )

    fig.show()


def calc_monthly_annual_rets(focus_df: pd.DataFrame, initial_balance: float):
    """
    this works for MVO
    where the portfolio is one continuous cumulative time series
    from the initial value of 100k up to whatever it ends up at
    for DRL on the other hand, we have one new portfolio for each year
    ie the portfolio value is reset to 100k at the start of each year
    so we need to calculate the returns for each year separately
    using a different function
    """
    # === Monthly returns ===
    month_df = focus_df[["portfolio_value"]].resample("ME").last()
    month_df = pd.concat(
        [
            pd.DataFrame(
                {"portfolio_value": [initial_balance]},
                index=[month_df.index.min() - pd.DateOffset(months=1)],
            ),
            month_df,
        ]
    )
    month_df["monthly_ret"] = month_df["portfolio_value"].pct_change() * 100
    month_df = month_df.iloc[1:]
    month_df["year"] = month_df.index.year
    month_df["month"] = month_df.index.month
    monthly_pivot = month_df.pivot(index="year", columns="month", values="monthly_ret")
    monthly_pivot = monthly_pivot.sort_index(ascending=True)

    # === Annual returns ===
    annual_df = focus_df[["portfolio_value"]].resample("YE").last()
    annual_df = pd.concat(
        [
            pd.DataFrame(
                {"portfolio_value": [initial_balance]},
                index=[annual_df.index.min() - pd.DateOffset(years=1)],
            ),
            annual_df,
        ]
    )
    annual_df["annual_ret"] = annual_df["portfolio_value"].pct_change() * 100
    annual_df = annual_df.iloc[1:]
    annual_df["year"] = annual_df.index.year
    annual_df = annual_df.sort_values(by="year")

    return monthly_pivot, annual_df


def plot_drl_portfolios_interactive(df_drl_port: pd.DataFrame):
    # plot portfolio value development with hue = year
    # plt.figure(figsize=(15, 5))
    # sns.lineplot(x="trading_day", y="portfolio_value", hue="year", data=df_drl_port)
    # plt.axhline(y=100_000, color="grey", linestyle="--")
    # plt.show()

    # color_sequence = diverging.RdBu
    color_sequence = sequential.Viridis

    fig = px.line(
        df_drl_port,
        x="trading_day",
        y="portfolio_value",
        color="year",
        color_discrete_sequence=color_sequence,
        title="Portfolio Value Development by Year",
    )

    # Add horizontal line at y=100_000
    fig.add_hline(
        y=100_000,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Value",
        annotation_position="top left",
    )

    fig.update_layout(
        height=600,
        width=1200,
        xaxis_title="Trading Day",
        yaxis_title="Portfolio Value",
        legend_title="Year",
    )

    fig.show()


def process_drl_portfolios(drl_port_df: pd.DataFrame, initial_balance: int = 100_000):
    """
    this works input df containing all DRL portfolios concatenated
    each portfolio is for one testing year and starts at 100k
    so we need to calculate the returns for each year separately
    and then combine them into a single time series

    the input df has columns:
    - date
    - portfolio_value
    - cash
    - all the weights in format w_{ticker}
    - all the shares in format s_{ticker}
    but we only really need date and portfolio_value

    the monthly_pivot output will be a df with
    years as rows and each months as a separate columns

    the annual_df output will be a df with
    last date of each year as rows
    and columns : annual_ret, year, portfolio_value ( final )
    """

    df = drl_port_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["year"] = df.index.year

    monthly_records = []
    annual_records = []

    for year, group in df.groupby("year"):
        group = group.sort_index()

        # === Monthly Returns ===
        month_end = group["portfolio_value"].resample("ME").last()
        month_start = pd.Series(
            [initial_balance], index=[month_end.index.min() - pd.offsets.MonthEnd(1)]
        )
        monthly = pd.concat([month_start, month_end])
        monthly_ret = monthly.pct_change().iloc[1:] * 100

        for idx, ret in monthly_ret.items():
            monthly_records.append(
                {"year": year, "month": idx.month, "monthly_ret": ret}
            )

        # === Annual Return ===
        final_value = group["portfolio_value"].iloc[-1]
        annual_ret = (final_value / initial_balance - 1) * 100
        annual_records.append(
            {
                "year": year,
                "portfolio_value": final_value,
                "annual_ret": annual_ret,
                "date": group.index[-1],
            }
        )

    # Construct outputs
    monthly_df = pd.DataFrame(monthly_records)
    monthly_pivot = monthly_df.pivot(
        index="year", columns="month", values="monthly_ret"
    ).sort_index()

    annual_df = pd.DataFrame(annual_records).set_index("date").sort_index()

    return monthly_pivot, annual_df


def plot_fig2(mvo_metrics_df: pd.DataFrame, save_dir: str, fname: str):
    sns.reset_defaults()
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Plot key metrics
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    # fig.suptitle("Portfolio Metrics by Lookback Period")

    # Sharpe Ratio
    sns.lineplot(data=mvo_metrics_df, x="lookback", y="Sharpe ratio", ax=axes[0])
    axes[0].set_title("Sharpe Ratio")

    # Max Drawdown
    sns.lineplot(data=mvo_metrics_df, x="lookback", y="Max drawdown", ax=axes[1])
    axes[1].set_title("Maximum Drawdown")

    # Avg Daily Change in Portfolio Weights / Turnover ?
    sns.lineplot(
        data=mvo_metrics_df, x="lookback", y="Avg Annual Turnover (in %)", ax=axes[2]
    )
    axes[2].set_title("Average Turnover")

    # for all axes, plot vline at lookback = 60
    for ax in axes:
        ax.axvline(x=60, color="r", linestyle="--", linewidth=1, label="lookback=60")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{fname}.pdf", dpi=300)
    plt.show()


def plot_fig4(monthly_pivot, annual_df, save_dir, fname):
    monthly_rets = monthly_pivot.values.reshape(-1)
    annual_mean = annual_df["annual_ret"].mean()

    # === PLOT with square plots and updated formatting ===
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Monthly heatmap
    sns.heatmap(
        monthly_pivot,
        cmap="RdYlGn",
        center=0,
        annot=True,
        annot_kws={"size": 11},
        fmt=".1f",
        ax=axes[0],
        cbar=False,
        linewidths=0.2,
        linecolor="white",
        # square=True,
    )
    axes[0].set_title("Monthly returns (%)")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Year")
    # axes[0].set_aspect("equal")

    # 2. Annual returns bar chart (scaled to %)
    axes[1].barh(annual_df["year"], annual_df["annual_ret"], color="steelblue")
    axes[1].axvline(
        x=annual_mean,
        color="skyblue",
        linestyle="--",
        lw=3,
        label=f"Mean = {annual_mean:.2f}%",
    )
    axes[1].set_title("Annual returns")
    axes[1].set_yticks(annual_df["year"])
    axes[1].invert_yaxis()

    # 3. Monthly return distribution
    axes[2].hist(monthly_rets, bins=20, color="#ff5812", edgecolor="white")
    axes[2].axvline(
        x=monthly_rets.mean(),
        color="gold",
        linestyle="--",
        lw=3,
        label=f"Mean = {monthly_rets.mean():.2f}%",
    )
    axes[2].set_title("Distribution of monthly returns")
    axes[2].set_ylabel("Number of months")

    # Formatting for bar charts and hist
    for idx in [1, 2]:
        axes[idx].axvline(x=0, color="black", lw=2)
        axes[idx].set_xlabel("Returns (%)")
        axes[idx].xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        axes[idx].legend(loc="upper right")

    # Super title and layout
    # plt.suptitle(f"Performance of MVO strategy with lookback = {lookback}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{fname}.pdf", dpi=300)
    plt.show()


def calc_portfolio_metrics(df: pd.DataFrame, risk_free_rate: float = 0):
    """
    Calculate comprehensive portfolio performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'date' and 'portfolio_value' columns.
        Optional: 'purchases' and 'sales' for turnover.
    risk_free_rate : float, optional
        Annualized risk-free rate used in Sharpe, Sortino, Omega ratio.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
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
        tail_ratio = (
            np.inf
            if abs(percentile_5) < 1e-9 and percentile_95 > 1e-9
            else (1 if abs(percentile_5) < 1e-9 else percentile_95 / abs(percentile_5))
        )

    # Stability
    stability = 1 / (1 + annual_volatility) if pd.notna(annual_volatility) else np.nan

    # Portfolio turnover
    if "purchases" in df.columns and "sales" in df.columns:
        df["year"] = df["date"].dt.year
        turnovers = []
        for year, group in df.groupby("year"):
            purchases = group["purchases"].sum()
            sales = group["sales"].sum()
            avg_value = group["portfolio_value"].mean()
            turnover = min(purchases, sales) / avg_value if avg_value > 0 else np.nan
            turnovers.append(turnover)
        annual_turnover = np.nanmean(turnovers)
    else:
        annual_turnover = np.nan

    return {
        # returns
        "Annual Returns": annual_return,
        "Cumulative Returns": cumulative_return,
        "Stability": stability,
        # risk
        "Annual Volatility": annual_volatility,
        "Max Drawdown": max_drawdown,
        "Daily VaR (95%)": var_95,
        "Tail Ratio": tail_ratio,
        # risk-adjusted returns
        "Sharpe Ratio": sharpe_ratio,
        "Calmar Ratio": calmar_ratio,
        "Omega Ratio": omega_ratio,
        "Sortino Ratio": sortino_ratio,
        # others
        "Skew": skew,
        "Kurtosis": kurtosis,
        "Turnover": annual_turnover,
    }


def compute_strategy_metrics(
    mvo_df,
    naive_df,
    buy_n_hold_df,
    drl_dfs: dict[str, pd.DataFrame],
    lookbacks,
):
    """
    Compute average yearly portfolio metrics for multiple strategies and lookback periods.

    Parameters:
    - mvo_df: DataFrame with a 'lookback' column and datetime index
    - naive_df: naive portfolio DataFrame with datetime index
    - buy_n_hold_df: buy and hold DataFrame with a 'Date' column
    - drl_df: DRL strategy DataFrame with datetime index
    - lookbacks: list of integers specifying lookback periods to evaluate
    - eval_n_plot: module or object with `calc_portfolio_metrics(df)` method

    Returns:
    - metrics_summary_df: DataFrame where each column is a strategy/lookback combination
    """

    results = {}

    # DRL strategy
    for model_name, df in drl_dfs.items():
        metrics = [calc_portfolio_metrics(df_yr) for _, df_yr in df.groupby("year")]
        results[f"DRL_{model_name}"] = pd.DataFrame(metrics).mean()

    for lb in lookbacks:
        # MVO
        df = (
            mvo_df[mvo_df["lookback"] == lb]
            .reset_index()
            .rename(columns={"index": "date"})
            .copy()
        )
        df["year"] = df["date"].dt.year
        metrics = [calc_portfolio_metrics(df_yr) for _, df_yr in df.groupby("year")]
        results[f"MVO_{lb}"] = pd.DataFrame(metrics).mean()

    # Naive portfolio (only one version, no lookback variation)
    df = naive_df.copy().reset_index().rename(columns={"index": "date"})
    df["year"] = df["date"].dt.year
    metrics = [calc_portfolio_metrics(df_yr) for _, df_yr in df.groupby("year")]
    results["Naive"] = pd.DataFrame(metrics).mean()

    # Buy & Hold
    df = buy_n_hold_df.copy().reset_index().rename(columns={"Date": "date"})
    df["year"] = df["date"].dt.year
    metrics = [calc_portfolio_metrics(df_yr) for _, df_yr in df.groupby("year")]
    results["Buy_n_Hold"] = pd.DataFrame(metrics).mean()

    # Combine results into single DataFrame
    metrics_summary_df = pd.DataFrame(results)

    return metrics_summary_df
