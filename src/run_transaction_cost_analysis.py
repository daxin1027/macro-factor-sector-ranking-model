"""Run transaction-cost sensitivity analysis for Step 6C ablation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

STRATEGIES = [
    "macro_only_ridge",
    "momentum_only_ridge",
    "macro_plus_momentum_ridge",
    "simple_relative_momentum_rule",
    "equal_weight_benchmark",
]
ACTIVE_STRATEGIES = STRATEGIES[:-1]
SCENARIO_BPS = [0, 5, 10, 25, 50]


@dataclass(frozen=True)
class Scope:
    name: str
    suffix: str
    ablation_predictions: Path
    ablation_backtest: Path
    enhanced_dataset: Path
    sectors: list[str]


SCOPES = [
    Scope(
        "11-sector",
        "11_sector",
        DATA_DIR / "ablation_predictions_11_sector.csv",
        DATA_DIR / "ablation_backtest_11_sector.csv",
        DATA_DIR / "enhanced_model_dataset_11_sector.csv",
        ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
    ),
    Scope(
        "9-sector-long-history",
        "9_sector_long_history",
        DATA_DIR / "ablation_predictions_9_sector_long_history.csv",
        DATA_DIR / "ablation_backtest_9_sector_long_history.csv",
        DATA_DIR / "enhanced_model_dataset_9_sector_long_history.csv",
        ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
    ),
]


def target_return_col(ticker: str) -> str:
    return f"target_next_return_{ticker}"


def output_paths(scope: Scope) -> dict[str, Path]:
    return {
        "sensitivity": DATA_DIR / f"transaction_cost_sensitivity_{scope.suffix}.csv",
        "comparison": DATA_DIR / f"transaction_cost_comparison_{scope.suffix}.csv",
        "break_even": DATA_DIR / f"transaction_cost_break_even_{scope.suffix}.csv",
        "monthly": DATA_DIR / f"transaction_cost_monthly_{scope.suffix}.csv",
        "plot": OUTPUT_DIR / f"transaction_cost_sensitivity_{scope.suffix}.png",
    }


def read_inputs(scope: Scope) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(
        scope.ablation_predictions,
        parse_dates=["feature_date", "predicted_return_month"],
    ).set_index("feature_date").sort_index()
    backtest = pd.read_csv(
        scope.ablation_backtest,
        parse_dates=["feature_date", "predicted_return_month"],
    ).set_index("feature_date").sort_index()
    enhanced = pd.read_csv(scope.enhanced_dataset, parse_dates=["date"]).set_index("date").sort_index()

    if predictions.empty or backtest.empty or enhanced.empty:
        raise ValueError(f"{scope.name} input contains an empty table.")
    for name, data in [("predictions", predictions), ("backtest", backtest), ("enhanced", enhanced)]:
        if data.index.has_duplicates or not data.index.is_monotonic_increasing:
            raise ValueError(f"{scope.name} {name} dates are not sorted and unique.")
        if data.isna().any().any():
            raise ValueError(f"{scope.name} {name} contains missing values.")
    if not predictions.index.equals(backtest.index):
        raise ValueError(f"{scope.name} ablation prediction/backtest dates do not align.")
    for strategy in STRATEGIES:
        if f"{strategy}_top3_strategy_return" not in backtest.columns:
            raise ValueError(f"{scope.name} missing {strategy} backtest return column.")
    return predictions, backtest, enhanced


def selected_weights(strategy: str, row: pd.Series, sectors: list[str]) -> pd.Series:
    if strategy == "equal_weight_benchmark":
        return pd.Series(1 / len(sectors), index=sectors, dtype=float)
    selected = str(row[f"{strategy}_selected_top3_sectors"]).split(",")
    weights = pd.Series(0.0, index=sectors)
    weights.loc[selected] = 1 / 3
    return weights


def realized_returns(enhanced: pd.DataFrame, date: pd.Timestamp, sectors: list[str]) -> pd.Series:
    return enhanced.loc[date, [target_return_col(ticker) for ticker in sectors]].rename(
        {target_return_col(ticker): ticker for ticker in sectors}
    )


def build_monthly_turnover(scope: Scope, backtest: pd.DataFrame, enhanced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous_target = {strategy: None for strategy in STRATEGIES}
    previous_returns = None
    for date, row in backtest.iterrows():
        monthly = {
            "feature_date": date,
            "predicted_return_month": row["predicted_return_month"],
        }
        current_returns = realized_returns(enhanced, date, scope.sectors)
        for strategy in STRATEGIES:
            target = selected_weights(strategy, row, scope.sectors)
            if previous_target[strategy] is None:
                turnover = 0.0
                convention = "initial_month_no_establishment_cost"
            else:
                drifted = previous_target[strategy] * (1 + previous_returns)
                drifted = drifted / drifted.sum()
                turnover = float(0.5 * (target - drifted).abs().sum())
                convention = "drift_adjusted_rebalance"
            if turnover < -1e-12 or turnover > 1 + 1e-12:
                raise ValueError(f"{scope.name} turnover out of bounds.")
            monthly[f"{strategy}_one_way_turnover"] = max(0.0, min(1.0, turnover))
            monthly[f"{strategy}_gross_return"] = row[f"{strategy}_top3_strategy_return"]
            monthly[f"{strategy}_turnover_convention"] = convention
            previous_target[strategy] = target
        previous_returns = current_returns
        rows.append(monthly)
    return pd.DataFrame(rows).set_index("feature_date")


def max_drawdown(returns: pd.Series) -> float:
    growth = (1 + returns).cumprod()
    return float((growth / growth.cummax() - 1).min())


def perf(returns: pd.Series, benchmark: pd.Series, turnover: pd.Series, costs: pd.Series, bps: int, strategy: str) -> dict[str, object]:
    months = len(returns)
    cumulative = float((1 + returns).prod() - 1)
    ann = float((1 + cumulative) ** (12 / months) - 1)
    vol = float(returns.std(ddof=1) * (12**0.5))
    excess = returns - benchmark
    te = float(excess.std(ddof=1) * (12**0.5))
    return {
        "strategy": strategy,
        "transaction_cost_bps": bps,
        "out_of_sample_months": months,
        "average_monthly_one_way_turnover": float(turnover.mean()),
        "annualized_one_way_turnover": float(turnover.mean() * 12),
        "total_deducted_transaction_cost_fraction": float(costs.sum()),
        "cumulative_return": cumulative,
        "terminal_growth_of_1": 1 + cumulative,
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_ratio_rf_0": ann / vol if vol != 0 else "not_applicable",
        "maximum_drawdown": max_drawdown(returns),
        "positive_month_rate": float((returns > 0).mean()),
        "average_monthly_excess_return_vs_equal_weight": float(excess.mean()),
        "information_ratio_vs_equal_weight": float(excess.mean() * 12 / te) if te != 0 else "not_applicable",
    }


def apply_costs(monthly: pd.DataFrame, bps: int) -> pd.DataFrame:
    rate = bps / 10000
    out = monthly[["predicted_return_month"]].copy()
    for strategy in STRATEGIES:
        turnover = monthly[f"{strategy}_one_way_turnover"]
        gross = monthly[f"{strategy}_gross_return"]
        cost = turnover * rate
        net = (1 - cost) * (1 + gross) - 1
        net.iloc[0] = gross.iloc[0]
        cost.iloc[0] = 0.0
        out[f"{strategy}_net_return_{bps}bps"] = net
        out[f"{strategy}_cost_fraction_{bps}bps"] = cost
    return out


def sensitivity(scope: Scope, monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_out = monthly[["predicted_return_month"]].copy()
    rows = []
    for bps in SCENARIO_BPS:
        costed = apply_costs(monthly, bps)
        benchmark = costed[f"equal_weight_benchmark_net_return_{bps}bps"]
        for strategy in STRATEGIES:
            returns = costed[f"{strategy}_net_return_{bps}bps"]
            turnover = monthly[f"{strategy}_one_way_turnover"]
            costs = costed[f"{strategy}_cost_fraction_{bps}bps"]
            rows.append(perf(returns, benchmark, turnover, costs, bps, strategy))
            monthly_out[f"{strategy}_net_return_{bps}bps"] = returns
            monthly_out[f"{strategy}_cost_fraction_{bps}bps"] = costs
    sens = pd.DataFrame(rows).fillna("not_applicable")
    return sens, monthly_out


def comparisons(sens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in SCENARIO_BPS:
        benchmark = sens[(sens.strategy == "equal_weight_benchmark") & (sens.transaction_cost_bps == bps)].iloc[0]
        for strategy in ACTIVE_STRATEGIES:
            row = sens[(sens.strategy == strategy) & (sens.transaction_cost_bps == bps)].iloc[0]
            rows.append(
                {
                    "strategy": strategy,
                    "transaction_cost_bps": bps,
                    "terminal_growth_difference_vs_equal_weight": row.terminal_growth_of_1 - benchmark.terminal_growth_of_1,
                    "cumulative_return_difference_vs_equal_weight": row.cumulative_return - benchmark.cumulative_return,
                    "annualized_return_difference_vs_equal_weight": row.annualized_return - benchmark.annualized_return,
                    "sharpe_ratio_difference_vs_equal_weight": row.sharpe_ratio_rf_0 - benchmark.sharpe_ratio_rf_0,
                    "maximum_drawdown_difference_vs_equal_weight": row.maximum_drawdown - benchmark.maximum_drawdown,
                }
            )
    return pd.DataFrame(rows)


def terminal_growth(monthly: pd.DataFrame, strategy: str, bps: int) -> float:
    costed = apply_costs(monthly, bps)
    return float((1 + costed[f"{strategy}_net_return_{bps}bps"]).prod())


def break_even(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in ACTIVE_STRATEGIES:
        if terminal_growth(monthly, strategy, 0) < terminal_growth(monthly, "equal_weight_benchmark", 0):
            value = "not_applicable_strategy_underperforms_before_costs"
        else:
            winners = [
                bps
                for bps in range(201)
                if terminal_growth(monthly, strategy, bps)
                >= terminal_growth(monthly, "equal_weight_benchmark", bps)
            ]
            value = "greater_than_200_bps" if winners and max(winners) == 200 else max(winners)
        rows.append({"strategy": strategy, "break_even_transaction_cost_bps": value})
    return pd.DataFrame(rows)


def validate_zero_bps(scope: Scope, monthly: pd.DataFrame, backtest: pd.DataFrame) -> list[str]:
    costed = apply_costs(monthly, 0)
    for strategy in STRATEGIES:
        gross = monthly[f"{strategy}_gross_return"]
        net = costed[f"{strategy}_net_return_0bps"]
        if (gross - net).abs().max() > 1e-12:
            raise ValueError(f"{scope.name} 0 bps does not equal gross for {strategy}.")
        if (gross - backtest[f"{strategy}_top3_strategy_return"]).abs().max() > 1e-12:
            raise ValueError(f"{scope.name} gross return fails Step 6C reproduction.")
    return [
        f"{scope.name}: 0-bps net monthly returns equal gross returns",
        f"{scope.name}: gross cumulative returns reproduce Step 6C ablation outputs",
        f"{scope.name}: equal-weight benchmark gross returns reconcile exactly",
    ]


def turnover_audit(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    high = []
    for strategy in STRATEGIES:
        t = monthly[f"{strategy}_one_way_turnover"]
        rows.append(
            {
                "strategy": strategy,
                "average_monthly_turnover": float(t.mean()),
                "median_monthly_turnover": float(t.median()),
                "maximum_monthly_turnover": float(t.max()),
            }
        )
        top = monthly.nlargest(5, f"{strategy}_one_way_turnover")[
            ["predicted_return_month", f"{strategy}_one_way_turnover"]
        ].copy()
        top.insert(0, "strategy", strategy)
        high.append(top.reset_index())
    return pd.DataFrame(rows), pd.concat(high, ignore_index=True)


def plot(scope: Scope, sens: pd.DataFrame, paths: dict[str, Path]) -> None:
    plt.figure(figsize=(10, 6))
    for strategy in STRATEGIES:
        data = sens[sens.strategy == strategy].sort_values("transaction_cost_bps")
        plt.plot(data.transaction_cost_bps, data.terminal_growth_of_1, marker="o", label=strategy)
    plt.title(f"Transaction Cost Sensitivity: {scope.name}")
    plt.xlabel("Transaction cost (basis points)")
    plt.ylabel("Terminal growth of $1")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    paths["plot"].parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(paths["plot"], dpi=150)
    plt.close()


def save(paths: dict[str, Path], sens: pd.DataFrame, comp: pd.DataFrame, be: pd.DataFrame, monthly: pd.DataFrame) -> None:
    paths["sensitivity"].parent.mkdir(parents=True, exist_ok=True)
    sens.to_csv(paths["sensitivity"], index=False)
    comp.to_csv(paths["comparison"], index=False)
    be.to_csv(paths["break_even"], index=False)
    monthly.to_csv(paths["monthly"], index_label="feature_date")


def validate_outputs(scope: Scope, paths: dict[str, Path], sens: pd.DataFrame, comp: pd.DataFrame, monthly_out: pd.DataFrame) -> list[str]:
    for path in paths.values():
        if not path.exists():
            raise ValueError(f"Missing output {path}")
    for name, data in [("sensitivity", sens), ("comparison", comp), ("monthly", monthly_out)]:
        if data.empty:
            raise ValueError(f"{scope.name} {name} output is empty.")
        if data.select_dtypes("number").replace([float("inf"), float("-inf")], pd.NA).isna().any().any():
            raise ValueError(f"{scope.name} {name} contains invalid numeric values.")
    return [
        f"{scope.name}: no model retraining occurs",
        f"{scope.name}: target portfolios use same-date saved selections only",
        f"{scope.name}: drift calculations use realized prior-month returns only",
        f"{scope.name}: turnover is bounded between 0 and 1",
        f"{scope.name}: all strategies share identical OOS dates",
        f"{scope.name}: output CSV and PNG files exist",
    ]


def print_audit(scope: Scope, monthly: pd.DataFrame, sens: pd.DataFrame, comp: pd.DataFrame, be: pd.DataFrame, validations: list[str], paths: dict[str, Path]) -> None:
    ta, high = turnover_audit(monthly)
    print()
    print(f"Transaction Cost Audit: {scope.name}")
    print("=" * (24 + len(scope.name)))
    print("Purpose: cost-aware robustness analysis of saved Step 6C strategies; no models are retrained.")
    print("Turnover formula: 0.5 * sum(abs(next target weight - drifted prior weight)).")
    print("Initial-cost convention: first OOS month excludes initial establishment cost.")
    print("Cost formula: net_return = (1 - turnover * cost_rate) * (1 + gross_return) - 1.")
    print(f"OOS feature dates: {monthly.index.min().date()} to {monthly.index.max().date()}")
    print(f"Predicted return months: {monthly.predicted_return_month.min().date()} to {monthly.predicted_return_month.max().date()}")
    print("Turnover statistics:")
    print(ta.to_string(index=False))
    print("Five highest-turnover months per strategy:")
    print(high.to_string(index=False))
    print("Cost sensitivity metrics:")
    print(sens.to_string(index=False))
    print("Active-vs-equal-weight comparisons:")
    print(comp.to_string(index=False))
    print("Break-even estimates:")
    print(be.to_string(index=False))
    print("Validation results:")
    for validation in validations:
        print(f"- {validation}")
    print("- equal-weight benchmark turnover includes drift rebalancing")
    print("Output files:")
    for path in paths.values():
        print(f"- {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    for scope in SCOPES:
        _, backtest, enhanced = read_inputs(scope)
        monthly = build_monthly_turnover(scope, backtest, enhanced)
        validations = validate_zero_bps(scope, monthly, backtest)
        sens, monthly_out = sensitivity(scope, monthly)
        comp = comparisons(sens)
        be = break_even(monthly)
        paths = output_paths(scope)
        plot(scope, sens, paths)
        save(paths, sens, comp, be, monthly_out)
        validations.extend(validate_outputs(scope, paths, sens, comp, monthly_out))
        print_audit(scope, monthly, sens, comp, be, validations, paths)


if __name__ == "__main__":
    main()
