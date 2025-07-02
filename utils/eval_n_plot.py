import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import plotly.express as px
import plotly.graph_objects as go


def plot_portfolio_val(
    START_DATE: str,
    END_DATE: str,
    mvo_history_df: pd.DataFrame,
    naive_portfolio_df: pd.DataFrame = None,
    ref_df: pd.DataFrame = None,
    ref_ticker: str = None,
    initial_shares: float = None,
    results_dir: str = None,
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
        ref_df[ref_ticker].loc[START_DATE:END_DATE] * initial_shares,
        c="k",
        label=ref_ticker,
        lw=1.5,
    )

    if title is not None:
        plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    # save plot
    plt.savefig(f"{results_dir}/mvo_portfolio_value.pdf", dpi=300)
    plt.show()


def plot_portfolio_val_interactive(
    START_DATE: str,
    END_DATE: str,
    mvo_history_df: pd.DataFrame,
    naive_portfolio_df: pd.DataFrame = None,
    ref_df: pd.DataFrame = None,
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
    spy_series = ref_df[ref_ticker].loc[START_DATE:END_DATE] * initial_shares
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


def process_drl_portfolios(drl_port_df: pd.DataFrame, initial_balance : int = 100_000):
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
        month_start = pd.Series([initial_balance], index=[month_end.index.min() - pd.offsets.MonthEnd(1)])
        monthly = pd.concat([month_start, month_end])
        monthly_ret = monthly.pct_change().iloc[1:] * 100

        for idx, ret in monthly_ret.items():
            monthly_records.append({"year": year, "month": idx.month, "monthly_ret": ret})

        # === Annual Return ===
        final_value = group["portfolio_value"].iloc[-1]
        annual_ret = (final_value / initial_balance - 1) * 100
        annual_records.append({
            "year": year,
            "portfolio_value": final_value,
            "annual_ret": annual_ret,
            "date": group.index[-1]
        })

    # Construct outputs
    monthly_df = pd.DataFrame(monthly_records)
    monthly_pivot = monthly_df.pivot(index="year", columns="month", values="monthly_ret").sort_index()

    annual_df = pd.DataFrame(annual_records).set_index("date").sort_index()

    return monthly_pivot, annual_df


def plot_fig2(mvo_metrics_df: pd.DataFrame, results_dir: str, fname: str):
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
    plt.savefig(f"{results_dir}/{fname}.pdf", dpi=300)
    plt.show()


def plot_fig4(monthly_pivot, annual_df, results_dir, fname):
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
    axes[1].axvline(x=annual_mean, color="skyblue", linestyle="--", lw=3, label=f"Mean = {annual_mean:.2f}%")
    axes[1].set_title("Annual returns")
    axes[1].set_yticks(annual_df["year"])
    axes[1].invert_yaxis()

    # 3. Monthly return distribution
    axes[2].hist(monthly_rets, bins=20, color="#ff5812", edgecolor="white")
    axes[2].axvline(
        x=monthly_rets.mean(), color="gold", linestyle="--", lw=3, label=f"Mean = {monthly_rets.mean():.2f}%"
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
    plt.savefig(f"{results_dir}/{fname}.pdf", dpi=300)
    plt.show()
