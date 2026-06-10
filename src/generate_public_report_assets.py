"""Generate public-facing report charts and summary tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "results_summary.md"

STRATEGIES = [
    "macro_only_ridge",
    "momentum_only_ridge",
    "macro_plus_momentum_ridge",
    "simple_relative_momentum_rule",
    "equal_weight_benchmark",
]
PLOT_LABELS = {
    "macro_only_ridge": "Macro-only Ridge",
    "momentum_only_ridge": "Momentum-only Ridge",
    "macro_plus_momentum_ridge": "Macro + momentum Ridge",
    "simple_relative_momentum_rule": "Simple relative momentum",
    "equal_weight_benchmark": "Equal-weight benchmark",
}
SCOPES = {
    "11-sector recent-regime": {
        "suffix": "11_sector",
        "title": "11-Sector Recent-Regime Ablation",
        "ablation_backtest": DATA_DIR / "ablation_backtest_11_sector.csv",
        "ablation_metrics": DATA_DIR / "ablation_metrics_11_sector.csv",
        "tc_sensitivity": DATA_DIR / "transaction_cost_sensitivity_11_sector.csv",
        "tc_break_even": DATA_DIR / "transaction_cost_break_even_11_sector.csv",
    },
    "9-sector long-history": {
        "suffix": "9_sector_long_history",
        "title": "9-Sector Long-History Ablation",
        "ablation_backtest": DATA_DIR / "ablation_backtest_9_sector_long_history.csv",
        "ablation_metrics": DATA_DIR / "ablation_metrics_9_sector_long_history.csv",
        "tc_sensitivity": DATA_DIR / "transaction_cost_sensitivity_9_sector_long_history.csv",
        "tc_break_even": DATA_DIR / "transaction_cost_break_even_9_sector_long_history.csv",
    },
}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required analysis output: {path}")
    data = pd.read_csv(path, **kwargs)
    if data.empty:
        raise ValueError(f"Input is empty: {path}")
    return data


def growth_with_initial(backtest: pd.DataFrame, strategy: str) -> tuple[pd.Series, pd.Series]:
    returns = backtest[f"{strategy}_top3_strategy_return"]
    start_date = backtest["predicted_return_month"].min() - pd.offsets.MonthEnd(1)
    dates = pd.concat(
        [pd.Series([start_date]), backtest["predicted_return_month"].reset_index(drop=True)],
        ignore_index=True,
    )
    growth = pd.concat(
        [pd.Series([1.0]), (1 + returns).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    return dates, growth


def plot_ablation_growth(scope_name: str, info: dict[str, object], output_path: Path) -> None:
    backtest = read_csv(
        info["ablation_backtest"],
        parse_dates=["feature_date", "predicted_return_month"],
    )
    start = backtest["predicted_return_month"].min().date()
    end = backtest["predicted_return_month"].max().date()

    plt.figure(figsize=(10, 6))
    for strategy in STRATEGIES:
        dates, growth = growth_with_initial(backtest, strategy)
        plt.plot(dates, growth, label=PLOT_LABELS[strategy], linewidth=2)
    plt.title(f"{info['title']}: Growth of $1\nOOS return months: {start} to {end}")
    plt.xlabel("Return month")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_transaction_cost(scope_name: str, info: dict[str, object], output_path: Path) -> None:
    sensitivity = read_csv(info["tc_sensitivity"])
    selected = [
        "macro_plus_momentum_ridge",
        "macro_only_ridge",
        "momentum_only_ridge",
        "equal_weight_benchmark",
    ]
    plt.figure(figsize=(10, 6))
    for strategy in selected:
        data = sensitivity[sensitivity["strategy"] == strategy].sort_values(
            "transaction_cost_bps"
        )
        plt.plot(
            data["transaction_cost_bps"],
            data["terminal_growth_of_1"],
            marker="o",
            linewidth=2,
            label=PLOT_LABELS[strategy],
        )
    plt.title(f"{info['title']}: Transaction-Cost Sensitivity")
    plt.xlabel("Transaction cost per one-way turnover (bps)")
    plt.ylabel("Terminal growth of $1")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def model_comparison_summary() -> pd.DataFrame:
    rows = []
    for scope_name, info in SCOPES.items():
        backtest = read_csv(
            info["ablation_backtest"],
            parse_dates=["feature_date", "predicted_return_month"],
        )
        metrics = read_csv(info["ablation_metrics"])
        sensitivity = read_csv(info["tc_sensitivity"])
        turnover = sensitivity[sensitivity["transaction_cost_bps"] == 0].set_index("strategy")
        date_range = (
            f"{backtest['predicted_return_month'].min().date()} to "
            f"{backtest['predicted_return_month'].max().date()}"
        )
        for strategy in STRATEGIES:
            metric = metrics[metrics["strategy"] == strategy].iloc[0]
            turn = turnover.loc[strategy]
            rows.append(
                {
                    "scope": scope_name,
                    "OOS date range": date_range,
                    "OOS months": int(metric["month_count"]),
                    "strategy": strategy,
                    "cumulative return": metric["cumulative_return"],
                    "terminal growth of $1": metric["terminal_growth_of_1"],
                    "annualized return": metric["annualized_return"],
                    "annualized volatility": metric["annualized_volatility"],
                    "Sharpe ratio": metric["sharpe_ratio"],
                    "maximum drawdown": metric["maximum_drawdown"],
                    "positive-month rate": metric["positive_month_rate"],
                    "average monthly one-way turnover": turn[
                        "average_monthly_one_way_turnover"
                    ],
                    "annualized one-way turnover": turn["annualized_one_way_turnover"],
                }
            )
    return pd.DataFrame(rows)


def transaction_cost_highlights() -> pd.DataFrame:
    rows = []
    highlights = {
        "11-sector recent-regime": ["macro_plus_momentum_ridge", "equal_weight_benchmark"],
        "9-sector long-history": [
            "momentum_only_ridge",
            "macro_plus_momentum_ridge",
            "equal_weight_benchmark",
        ],
    }
    for scope_name, info in SCOPES.items():
        sensitivity = read_csv(info["tc_sensitivity"])
        break_even = read_csv(info["tc_break_even"]).set_index("strategy")
        for strategy in highlights[scope_name]:
            for bps in [0, 10, 25, 50]:
                row = sensitivity[
                    (sensitivity["strategy"] == strategy)
                    & (sensitivity["transaction_cost_bps"] == bps)
                ].iloc[0]
                rows.append(
                    {
                        "scope": scope_name,
                        "strategy": strategy,
                        "transaction-cost bps": bps,
                        "cumulative return": row["cumulative_return"],
                        "terminal growth of $1": row["terminal_growth_of_1"],
                        "annualized return": row["annualized_return"],
                        "Sharpe ratio": row["sharpe_ratio_rf_0"],
                        "maximum drawdown": row["maximum_drawdown"],
                        "break-even cost label or value": break_even.loc[strategy][
                            "break_even_transaction_cost_bps"
                        ]
                        if strategy in break_even.index
                        else "not_applicable_benchmark",
                    }
                )
    return pd.DataFrame(rows)


def sector_universe() -> pd.DataFrame:
    sectors = [
        ("XLB", "Materials", True, True),
        ("XLC", "Communication Services", True, False),
        ("XLE", "Energy", True, True),
        ("XLF", "Financials", True, True),
        ("XLI", "Industrials", True, True),
        ("XLK", "Technology", True, True),
        ("XLP", "Consumer Staples", True, True),
        ("XLRE", "Real Estate", True, False),
        ("XLU", "Utilities", True, True),
        ("XLV", "Health Care", True, True),
        ("XLY", "Consumer Discretionary", True, True),
    ]
    return pd.DataFrame(
        sectors,
        columns=[
            "ticker",
            "sector_name",
            "included_in_11_sector_scope",
            "included_in_9_sector_long_history_scope",
        ],
    )


def macro_dictionary() -> pd.DataFrame:
    rows = [
        ("indpro_yoy", "INDPRO", "12-month percentage change * 100", "Shifted by 1 calendar month", "Industrial activity growth"),
        ("unemployment_change_3m", "UNRATE", "3-month difference", "Shifted by 1 calendar month", "Labor-market deterioration or improvement"),
        ("inflation_yoy", "CPIAUCSL", "12-month percentage change * 100", "Shifted by 1 calendar month", "Consumer inflation pressure"),
        ("m2_yoy", "M2SL", "12-month percentage change * 100", "Shifted by 1 calendar month", "Money-supply growth"),
        ("fedfunds_level", "FEDFUNDS", "Level", "Shifted by 1 calendar month", "Short-rate policy stance"),
        ("treasury_10y_level", "GS10", "Level", "Shifted by 1 calendar month", "Long-rate level"),
        ("yield_curve_spread", "T10Y2Y", "Level", "Current completed month-end observation", "Yield-curve slope"),
        ("credit_spread", "BAA10Y", "Level", "Current completed month-end observation", "Corporate credit-risk compensation"),
        ("vix_level", "VIXCLS", "Level", "Current completed month-end observation", "Equity-market volatility/risk aversion"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "feature_name",
            "source_series",
            "transformation",
            "information_availability_rule",
            "interpretation",
        ],
    )


def write_summary() -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        """# Macro-Factor Sector Ranking Model: Results Summary

## Research Question

Can a small set of macroeconomic indicators and leakage-safe sector momentum features help rank U.S. sector ETFs for next-month relative performance?

## Data Scopes

The project evaluates two scopes. The 11-sector recent-regime scope includes all 11 U.S. sector ETFs but begins later because XLC and XLRE have shorter histories. The 9-sector long-history scope excludes XLC and XLRE to test robustness over a longer sample.

## Modeling Approach

The core model is an interpretable walk-forward Ridge Regression setup. Each sector gets a separate model, and each month sectors are ranked by predicted next-month returns. The project also compares macro-only, momentum-only, combined macro-plus-momentum, simple relative momentum, and equal-weight benchmark variants.

## Walk-Forward and Leakage Controls

Models use expanding-window training with no random train-test split. Training rows are strictly earlier than each prediction month. Macro features are lagged where appropriate, and sector momentum and volatility features use trailing completed monthly returns only. Date metadata, targets, and ranks are excluded from model inputs.

## Recent 11-Sector Result

In the recent 11-sector scope, the combined macro-plus-momentum model performed better than the macro-only and equal-weight benchmarks over the tested out-of-sample period. This result is promising but covers a short, recent market regime.

## Long-History 9-Sector Result

In the longer 9-sector scope, the combined model was not consistently better than equal weight. This weakens the evidence that the enhanced signal is broadly persistent across regimes.

## Transaction-Cost Conclusion

Transaction costs do not materially change the favorable recent-regime 11-sector conclusion for the combined model, but they reinforce the weaker long-history conclusion because active strategies already trail equal weight before costs in that scope.

## Limitations

This is not evidence of a deployable trading strategy. The results are regime-dependent, use simplified monthly rebalancing assumptions, and rely on standard FRED downloads that may include revised historical values. A future improvement would use ALFRED vintage data for stricter point-in-time macro feature construction.

## Overall Interpretation

The project shows that sector-specific momentum and volatility can improve a macro-only model in the recent 11-sector sample, but the long-history evidence is mixed to weak. The strongest conclusion is methodological: controlled walk-forward testing and same-row ablations are essential before interpreting apparent sector-ranking performance.
""",
        encoding="utf-8",
    )


def validate_public_files(paths: list[Path]) -> list[str]:
    validations = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty public asset: {path}")
        validations.append(f"{path.relative_to(PROJECT_ROOT)} exists and is non-empty")

    for table_path in TABLES_DIR.glob("*.csv"):
        text = table_path.read_text(encoding="utf-8")
        if "FRED_API_KEY" in text or "your_fred_api_key" in text:
            raise ValueError(f"Unsafe environment text found in {table_path}")
    validations.append("public tables contain no API keys or environment-variable values")
    validations.append("no model retraining occurs")
    validations.append("no modeling logic is changed")
    return validations


def main() -> None:
    ensure_dirs()

    figure_paths = [
        FIGURES_DIR / "01_recent_11_sector_ablation_growth.png",
        FIGURES_DIR / "02_long_history_9_sector_ablation_growth.png",
        FIGURES_DIR / "03_recent_11_sector_transaction_cost_sensitivity.png",
        FIGURES_DIR / "04_long_history_9_sector_transaction_cost_sensitivity.png",
    ]
    plot_ablation_growth(
        "11-sector recent-regime",
        SCOPES["11-sector recent-regime"],
        figure_paths[0],
    )
    plot_ablation_growth(
        "9-sector long-history",
        SCOPES["9-sector long-history"],
        figure_paths[1],
    )
    plot_transaction_cost(
        "11-sector recent-regime",
        SCOPES["11-sector recent-regime"],
        figure_paths[2],
    )
    plot_transaction_cost(
        "9-sector long-history",
        SCOPES["9-sector long-history"],
        figure_paths[3],
    )

    table_paths = [
        TABLES_DIR / "model_comparison_summary.csv",
        TABLES_DIR / "transaction_cost_highlights.csv",
        TABLES_DIR / "sector_etf_universe.csv",
        TABLES_DIR / "macro_feature_dictionary.csv",
    ]
    model_comparison_summary().to_csv(table_paths[0], index=False)
    transaction_cost_highlights().to_csv(table_paths[1], index=False)
    sector_universe().to_csv(table_paths[2], index=False)
    macro_dictionary().to_csv(table_paths[3], index=False)
    write_summary()

    validations = validate_public_files([*figure_paths, *table_paths, SUMMARY_PATH])

    print("Public Report Asset Generation Audit")
    print("====================================")
    print("Files created or refreshed:")
    for path in [*figure_paths, *table_paths, SUMMARY_PATH]:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    print()
    print("Public chart paths:")
    for path in figure_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    print()
    print("Public table paths:")
    for path in table_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    print()
    print("Validation results:")
    for validation in validations:
        print(f"- {validation}")


if __name__ == "__main__":
    main()
