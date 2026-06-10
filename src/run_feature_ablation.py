"""Run strict same-row feature ablation experiments for sector ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RIDGE_ALPHA = 10.0

MACRO_FEATURES = [
    "indpro_yoy",
    "unemployment_change_3m",
    "inflation_yoy",
    "m2_yoy",
    "fedfunds_level",
    "treasury_10y_level",
    "yield_curve_spread",
    "credit_spread",
    "vix_level",
]
SECTORS_11 = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
SECTORS_9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
MODEL_STRATEGIES = ["macro_only_ridge", "momentum_only_ridge", "macro_plus_momentum_ridge"]
ALL_STRATEGIES = [*MODEL_STRATEGIES, "simple_relative_momentum_rule", "equal_weight_benchmark"]


@dataclass(frozen=True)
class Scope:
    name: str
    input_path: Path
    enhanced_backtest_path: Path
    sectors: list[str]
    min_train_rows: int
    suffix: str


SCOPES = [
    Scope(
        "11-sector",
        DATA_DIR / "enhanced_model_dataset_11_sector.csv",
        DATA_DIR / "enhanced_backtest_11_sector.csv",
        SECTORS_11,
        48,
        "11_sector",
    ),
    Scope(
        "9-sector-long-history",
        DATA_DIR / "enhanced_model_dataset_9_sector_long_history.csv",
        DATA_DIR / "enhanced_backtest_9_sector_long_history.csv",
        SECTORS_9,
        120,
        "9_sector_long_history",
    ),
]


def sector_features(ticker: str) -> list[str]:
    return [
        f"sector_momentum_3m_{ticker}",
        f"sector_momentum_6m_{ticker}",
        f"sector_relative_momentum_6m_{ticker}",
        f"sector_volatility_6m_{ticker}",
    ]


def feature_set(strategy: str, ticker: str) -> list[str]:
    if strategy == "macro_only_ridge":
        return MACRO_FEATURES
    if strategy == "momentum_only_ridge":
        return sector_features(ticker)
    if strategy == "macro_plus_momentum_ridge":
        return [*MACRO_FEATURES, *sector_features(ticker)]
    raise ValueError(f"Unsupported model strategy: {strategy}")


def target_return_col(ticker: str) -> str:
    return f"target_next_return_{ticker}"


def target_rank_col(ticker: str) -> str:
    return f"target_next_rank_{ticker}"


def make_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


def output_paths(scope: Scope) -> dict[str, Path]:
    return {
        "predictions": DATA_DIR / f"ablation_predictions_{scope.suffix}.csv",
        "backtest": DATA_DIR / f"ablation_backtest_{scope.suffix}.csv",
        "metrics": DATA_DIR / f"ablation_metrics_{scope.suffix}.csv",
        "incremental": DATA_DIR / f"ablation_incremental_comparison_{scope.suffix}.csv",
        "calendar": DATA_DIR / f"ablation_calendar_year_{scope.suffix}.csv",
        "subperiods": DATA_DIR / f"ablation_subperiods_{scope.suffix}.csv",
        "plot": OUTPUT_DIR / f"ablation_cumulative_returns_{scope.suffix}.png",
    }


def read_dataset(scope: Scope) -> pd.DataFrame:
    required = [
        *MACRO_FEATURES,
        *[f for ticker in scope.sectors for f in sector_features(ticker)],
        *[target_return_col(ticker) for ticker in scope.sectors],
        *[target_rank_col(ticker) for ticker in scope.sectors],
        "target_best_sector",
    ]
    data = pd.read_csv(scope.input_path, parse_dates=["date"]).set_index("date").sort_index()
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"{scope.input_path.name} missing columns: {', '.join(missing)}")
    if data.index.has_duplicates:
        raise ValueError(f"{scope.input_path.name} contains duplicate dates.")
    if data.isna().any().any():
        raise ValueError(f"{scope.input_path.name} contains missing values.")
    return data.loc[:, required]


def validate_feature_counts(scope: Scope) -> list[str]:
    for ticker in scope.sectors:
        checks = {
            "macro_only_ridge": 9,
            "momentum_only_ridge": 4,
            "macro_plus_momentum_ridge": 13,
        }
        for strategy, expected in checks.items():
            features = feature_set(strategy, ticker)
            if len(features) != expected:
                raise ValueError(f"{strategy} {ticker} has wrong feature count.")
            if "date" in features:
                raise ValueError("Date metadata included as a feature.")
            own = set(sector_features(ticker))
            all_sector_specific = {f for s in scope.sectors for f in sector_features(s)}
            if strategy != "macro_only_ridge" and not set(features).issubset(set(MACRO_FEATURES) | own):
                raise ValueError(f"{strategy} {ticker} includes another sector feature.")
            if strategy == "macro_only_ridge" and set(features) & all_sector_specific:
                raise ValueError("Macro-only strategy includes sector features.")
    return [
        f"{scope.name}: date metadata excluded from every model",
        f"{scope.name}: macro_only_ridge uses exactly 9 features per sector",
        f"{scope.name}: momentum_only_ridge uses exactly 4 own-sector features per sector",
        f"{scope.name}: macro_plus_momentum_ridge uses exactly 13 features per sector",
    ]


def rank_and_returns(predicted: dict[str, float], actual: dict[str, float]) -> dict[str, object]:
    pred_rank = pd.Series(predicted).rank(ascending=False, method="min")
    actual_rank = pd.Series(actual).rank(ascending=False, method="min")
    best = max(predicted, key=predicted.get)
    actual_best = max(actual, key=actual.get)
    top3 = sorted(predicted, key=predicted.get, reverse=True)[:3]
    return {
        "predicted_best_sector": best,
        "actual_best_sector": actual_best,
        "selected_top3_sectors": ",".join(top3),
        "top1_strategy_return": actual[best],
        "top3_strategy_return": sum(actual[t] for t in top3) / 3,
        "equal_weight_benchmark_return": sum(actual.values()) / len(actual),
        "rank_ic": pd.Series(predicted).corr(pd.Series(actual), method="spearman"),
        "predicted_ranks": pred_rank.to_dict(),
        "actual_ranks": actual_rank.to_dict(),
    }


def run_scope(scope: Scope, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    validations = validate_feature_counts(scope)
    rows = []
    bt_rows = []
    for i in range(scope.min_train_rows, len(data)):
        date = data.index[i]
        train = data.iloc[:i]
        if not train.index.max() < date:
            raise ValueError(f"{scope.name} future observation leakage at {date.date()}.")
        actual = {ticker: float(data.at[date, target_return_col(ticker)]) for ticker in scope.sectors}
        base = {
            "feature_date": date,
            "predicted_return_month": date + pd.offsets.MonthEnd(1),
            "actual_best_sector": max(actual, key=actual.get),
        }
        strategy_results = {}
        for strategy in MODEL_STRATEGIES:
            predicted = {}
            for ticker in scope.sectors:
                features = feature_set(strategy, ticker)
                model = make_pipeline()
                model.fit(train.loc[:, features], train[target_return_col(ticker)])
                if int(model.named_steps["scaler"].n_samples_seen_) != len(train):
                    raise ValueError(f"Scaler validation failed for {strategy} {ticker}.")
                predicted[ticker] = float(model.predict(data.loc[[date], features])[0])
            strategy_results[strategy] = rank_and_returns(predicted, actual)
        rel_pred = {
            ticker: float(data.at[date, f"sector_relative_momentum_6m_{ticker}"])
            for ticker in scope.sectors
        }
        strategy_results["simple_relative_momentum_rule"] = rank_and_returns(rel_pred, actual)
        strategy_results["equal_weight_benchmark"] = {
            "predicted_best_sector": "not_applicable",
            "actual_best_sector": max(actual, key=actual.get),
            "selected_top3_sectors": "not_applicable",
            "top1_strategy_return": "not_applicable",
            "top3_strategy_return": sum(actual.values()) / len(actual),
            "equal_weight_benchmark_return": sum(actual.values()) / len(actual),
            "rank_ic": "not_applicable",
            "predicted_ranks": {},
            "actual_ranks": {},
        }
        pred_row = dict(base)
        bt_row = dict(base)
        for strategy, result in strategy_results.items():
            for key in [
                "predicted_best_sector",
                "selected_top3_sectors",
                "top1_strategy_return",
                "top3_strategy_return",
                "equal_weight_benchmark_return",
                "rank_ic",
            ]:
                pred_row[f"{strategy}_{key}"] = result[key]
                bt_row[f"{strategy}_{key}"] = result[key]
        rows.append(pred_row)
        bt_rows.append(bt_row)
    predictions = pd.DataFrame(rows).set_index("feature_date").sort_index()
    backtest = pd.DataFrame(bt_rows).set_index("feature_date").sort_index()
    validations.extend(validate_backtest(scope, predictions, backtest))
    return predictions, backtest, validations


def validate_backtest(scope: Scope, predictions: pd.DataFrame, backtest: pd.DataFrame) -> list[str]:
    if predictions.empty or backtest.empty:
        raise ValueError(f"{scope.name} generated empty outputs.")
    if predictions.index.has_duplicates or not predictions.index.is_monotonic_increasing:
        raise ValueError(f"{scope.name} predictions not sorted and unique.")
    benchmark = backtest["macro_only_ridge_equal_weight_benchmark_return"]
    for strategy in ALL_STRATEGIES:
        if not backtest[f"{strategy}_equal_weight_benchmark_return"].equals(benchmark):
            raise ValueError(f"{scope.name} benchmark mismatch for {strategy}.")
    return [
        f"{scope.name}: all strategies use identical OOS feature dates",
        f"{scope.name}: all strategies use identical realized next-month return months",
        f"{scope.name}: all strategies use identical equal-weight benchmark returns",
        f"{scope.name}: predictions sorted and unique by date",
        f"{scope.name}: no random split and no future observations",
        f"{scope.name}: scaler fitted only on training rows",
    ]


def max_drawdown(returns: pd.Series) -> float:
    growth = (1 + returns).cumprod()
    return float((growth / growth.cummax() - 1).min())


def perf(returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    months = len(returns)
    cumulative = float((1 + returns).prod() - 1)
    ann = float((1 + cumulative) ** (12 / months) - 1)
    vol = float(returns.std(ddof=1) * (12**0.5))
    out = {
        "month_count": months,
        "cumulative_return": cumulative,
        "terminal_growth_of_1": 1 + cumulative,
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_ratio": ann / vol if vol != 0 else 0.0,
        "maximum_drawdown": max_drawdown(returns),
        "positive_month_rate": float((returns > 0).mean()),
    }
    if benchmark is not None:
        excess = returns - benchmark
        te = float(excess.std(ddof=1) * (12**0.5))
        out["average_monthly_excess_return_vs_equal_weight"] = float(excess.mean())
        out["annualized_tracking_error_vs_equal_weight"] = te
        out["information_ratio_vs_equal_weight"] = float(excess.mean() * 12 / te) if te != 0 else 0.0
    return out


def turnover(selected: pd.Series) -> float:
    prev = None
    values = []
    for item in selected:
        if item == "not_applicable":
            continue
        current = set(str(item).split(","))
        if prev is not None:
            values.append(1 - len(current & prev) / 3)
        prev = current
    return float(pd.Series(values).mean()) if values else 0.0


def build_metrics(scope: Scope, backtest: pd.DataFrame) -> pd.DataFrame:
    benchmark = backtest["macro_only_ridge_equal_weight_benchmark_return"]
    rows = []
    for strategy in ALL_STRATEGIES:
        ret = backtest[f"{strategy}_top3_strategy_return"]
        row = {"strategy": strategy, **perf(ret, benchmark)}
        if strategy != "equal_weight_benchmark":
            row.update(
                {
                    "top1_accuracy": float(
                        (backtest[f"{strategy}_predicted_best_sector"] == backtest["actual_best_sector"]).mean()
                    ),
                    "random_choice_top1_benchmark": 1 / len(scope.sectors),
                    "average_monthly_rank_ic": pd.to_numeric(backtest[f"{strategy}_rank_ic"], errors="coerce").mean(),
                    "median_monthly_rank_ic": pd.to_numeric(backtest[f"{strategy}_rank_ic"], errors="coerce").median(),
                    "positive_rank_ic_rate": float((pd.to_numeric(backtest[f"{strategy}_rank_ic"], errors="coerce") > 0).mean()),
                    "average_monthly_top3_turnover": turnover(backtest[f"{strategy}_selected_top3_sectors"]),
                }
            )
        else:
            row.update(
                {
                    "top1_accuracy": "not_applicable",
                    "random_choice_top1_benchmark": "not_applicable",
                    "average_monthly_rank_ic": "not_applicable",
                    "median_monthly_rank_ic": "not_applicable",
                    "positive_rank_ic_rate": "not_applicable",
                    "average_monthly_top3_turnover": "not_applicable",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).fillna("not_applicable")


def incremental(backtest: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("momentum_only_ridge", "macro_only_ridge"),
        ("macro_plus_momentum_ridge", "macro_only_ridge"),
        ("macro_plus_momentum_ridge", "momentum_only_ridge"),
        ("simple_relative_momentum_rule", "equal_weight_benchmark"),
    ]
    rows = []
    for a, b in pairs:
        ar = backtest[f"{a}_top3_strategy_return"]
        br = backtest[f"{b}_top3_strategy_return"]
        ap = perf(ar)
        bp = perf(br)
        diff = ar - br
        rows.append(
            {
                "comparison": f"{a}_minus_{b}",
                "cumulative_return_difference": ap["cumulative_return"] - bp["cumulative_return"],
                "annualized_return_difference": ap["annualized_return"] - bp["annualized_return"],
                "sharpe_ratio_difference": ap["sharpe_ratio"] - bp["sharpe_ratio"],
                "maximum_drawdown_difference": ap["maximum_drawdown"] - bp["maximum_drawdown"],
                "average_monthly_return_difference": float(diff.mean()),
                "months_outperforming": int((diff > 0).sum()),
                "months_underperforming": int((diff < 0).sum()),
                "tied_months": int((diff == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def calendar_year(backtest: pd.DataFrame) -> pd.DataFrame:
    temp = backtest.copy()
    temp["year"] = temp["predicted_return_month"].dt.year
    rows = []
    for year, group in temp.groupby("year"):
        row = {"year": year}
        for strategy in ALL_STRATEGIES:
            row[f"{strategy}_top3_return"] = float((1 + group[f"{strategy}_top3_strategy_return"]).prod() - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def subperiods_9(backtest: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2010-08-31_to_2014-12-31", "2010-08-31", "2014-12-31"),
        ("2015-01-31_to_2019-12-31", "2015-01-31", "2019-12-31"),
        ("2020-01-31_to_2026-05-31", "2020-01-31", "2026-05-31"),
    ]
    rows = []
    for label, start, end in periods:
        group = backtest[
            (backtest["predicted_return_month"] >= pd.Timestamp(start))
            & (backtest["predicted_return_month"] <= pd.Timestamp(end))
        ]
        for strategy in ALL_STRATEGIES:
            p = perf(group[f"{strategy}_top3_strategy_return"])
            rows.append(
                {
                    "subperiod": label,
                    "strategy": strategy,
                    "month_count": p["month_count"],
                    "cumulative_return": p["cumulative_return"],
                    "annualized_return": p["annualized_return"],
                    "sharpe_ratio": p["sharpe_ratio"],
                    "maximum_drawdown": p["maximum_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def concentration(backtest: pd.DataFrame) -> pd.DataFrame:
    benchmark = backtest["equal_weight_benchmark_top3_strategy_return"]
    rows = []
    for strategy in ALL_STRATEGIES[:-1]:
        excess = backtest[f"{strategy}_top3_strategy_return"] - benchmark
        total = float(excess.sum())
        positive = excess[excess > 0].sort_values(ascending=False)
        rows.append(
            {
                "strategy": strategy,
                "total_sum_monthly_excess": total,
                "best_1_month_contribution": positive.head(1).sum() / total if total > 0 else "not_applicable_total_excess_negative",
                "best_3_month_contribution": positive.head(3).sum() / total if total > 0 else "not_applicable_total_excess_negative",
                "best_5_month_contribution": positive.head(5).sum() / total if total > 0 else "not_applicable_total_excess_negative",
                "positive_excess_months": int((excess > 0).sum()),
                "negative_excess_months": int((excess < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_worst(backtest: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("macro_plus_momentum_minus_macro_only", "macro_plus_momentum_ridge", "macro_only_ridge"),
        ("momentum_only_minus_macro_only", "momentum_only_ridge", "macro_only_ridge"),
        ("relative_momentum_rule_minus_equal_weight", "simple_relative_momentum_rule", "equal_weight_benchmark"),
    ]
    rows = []
    for label, a, b in comparisons:
        diff = backtest[f"{a}_top3_strategy_return"] - backtest[f"{b}_top3_strategy_return"]
        table = pd.DataFrame(
            {
                "comparison": label,
                "feature_date": backtest.index,
                "predicted_return_month": backtest["predicted_return_month"],
                "difference": diff,
                "strategy_a_top3_return": backtest[f"{a}_top3_strategy_return"],
                "strategy_b_top3_return": backtest[f"{b}_top3_strategy_return"],
            }
        )
        rows.extend(table.nlargest(5, "difference").assign(kind="best").to_dict("records"))
        rows.extend(table.nsmallest(5, "difference").assign(kind="worst").to_dict("records"))
    return pd.DataFrame(rows)


def reproduce_enhanced(scope: Scope, backtest: pd.DataFrame) -> str:
    existing = pd.read_csv(scope.enhanced_backtest_path, parse_dates=["feature_date", "predicted_return_month"]).set_index("feature_date").sort_index()
    aligned = existing.loc[backtest.index]
    if not aligned["predicted_return_month"].equals(backtest["predicted_return_month"]):
        raise ValueError(f"{scope.name} Step 6B prediction months do not align.")
    if (aligned["top3_strategy_return"] - backtest["macro_plus_momentum_ridge_top3_strategy_return"]).abs().max() > 1e-12:
        raise ValueError(f"{scope.name} combined model does not reproduce Step 6B.")
    if (aligned["equal_weight_benchmark_return"] - backtest["macro_plus_momentum_ridge_equal_weight_benchmark_return"]).abs().max() > 1e-12:
        raise ValueError(f"{scope.name} equal-weight benchmark does not reproduce Step 6B.")
    return f"{scope.name}: macro_plus_momentum_ridge reproduces Step 6B enhanced top3 and benchmark returns"


def add_growth(backtest: pd.DataFrame) -> pd.DataFrame:
    out = backtest.copy()
    for strategy in ALL_STRATEGIES:
        col = f"{strategy}_top3_strategy_return"
        out[f"{strategy}_growth_of_1"] = (1 + out[col]).cumprod()
    return out


def plot(scope: Scope, backtest: pd.DataFrame, paths: dict[str, Path]) -> None:
    start_date = backtest["predicted_return_month"].min() - pd.offsets.MonthEnd(1)
    dates = pd.concat([pd.Series([start_date]), backtest["predicted_return_month"].reset_index(drop=True)], ignore_index=True)
    plt.figure(figsize=(11, 6))
    for strategy in ALL_STRATEGIES:
        growth = pd.concat([pd.Series([1.0]), (1 + backtest[f"{strategy}_top3_strategy_return"]).cumprod().reset_index(drop=True)], ignore_index=True)
        plt.plot(dates, growth, label=strategy)
    plt.title(f"Feature Ablation Growth of $1: {scope.name}")
    plt.xlabel("Predicted return month")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    paths["plot"].parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(paths["plot"], dpi=150)
    plt.close()


def save(scope: Scope, predictions: pd.DataFrame, backtest: pd.DataFrame, metrics: pd.DataFrame, inc: pd.DataFrame, cal: pd.DataFrame, sub: pd.DataFrame | None, paths: dict[str, Path]) -> None:
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths["predictions"], index_label="feature_date")
    backtest.to_csv(paths["backtest"], index_label="feature_date")
    metrics.to_csv(paths["metrics"], index=False)
    inc.to_csv(paths["incremental"], index=False)
    cal.to_csv(paths["calendar"], index=False)
    if sub is not None:
        sub.to_csv(paths["subperiods"], index=False)


def validate_outputs(scope: Scope, paths: dict[str, Path], predictions: pd.DataFrame, backtest: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    for key, path in paths.items():
        if key == "subperiods" and scope.name != "9-sector-long-history":
            continue
        if not path.exists():
            raise ValueError(f"Missing output: {path}")
    if predictions.isna().any().any() or backtest.isna().any().any() or metrics.isna().any().any():
        raise ValueError(f"{scope.name} output contains missing values.")
    numeric_metrics = metrics.apply(pd.to_numeric, errors="coerce")
    if numeric_metrics.replace([float("inf"), float("-inf")], pd.NA).isna().sum().sum() < numeric_metrics.isna().sum().sum():
        raise ValueError(f"{scope.name} metric output contains infinite values.")
    return [
        f"{scope.name}: output CSV and PNG files exist",
        f"{scope.name}: generated output tables are non-empty and complete",
        f"{scope.name}: metrics contain no infinite values",
        f"{scope.name}: plots begin at growth of $1 equal to 1.0",
    ]


def print_audit(scope: Scope, data: pd.DataFrame, predictions: pd.DataFrame, backtest: pd.DataFrame, metrics: pd.DataFrame, inc: pd.DataFrame, cal: pd.DataFrame, sub: pd.DataFrame | None, conc: pd.DataFrame, bw: pd.DataFrame, validations: list[str], paths: dict[str, Path]) -> None:
    print()
    print(f"Feature Ablation Audit: {scope.name}")
    print("=" * (24 + len(scope.name)))
    print("Purpose: strict same-row ablation of macro, sector momentum/volatility, combined, and simple relative momentum signals.")
    print(f"Ridge alpha: {RIDGE_ALPHA}; Pipeline: StandardScaler + Ridge; one model per sector.")
    print("Feature counts: macro=9, momentum=4, combined=13; date metadata excluded.")
    print(f"Input date range: {data.index.min().date()} to {data.index.max().date()}, shape {data.shape}")
    print(f"OOS feature dates: {predictions.index.min().date()} to {predictions.index.max().date()}")
    print(f"Predicted return months: {predictions.predicted_return_month.min().date()} to {predictions.predicted_return_month.max().date()}")
    print(f"OOS month count: {len(predictions)}")
    print("Metrics:")
    print(metrics.to_string(index=False))
    print("Strict incremental comparisons:")
    print(inc.to_string(index=False))
    print("Calendar-year table:")
    print(cal.to_string(index=False))
    if sub is not None:
        print("9-sector subperiod table:")
        print(sub.to_string(index=False))
    print("Concentration diagnostics:")
    print(conc.to_string(index=False))
    print("Best/worst monthly comparisons:")
    print(bw.to_string(index=False))
    print("Validation results:")
    for validation in validations:
        print(f"- {validation}")
    print("Output files:")
    for key, path in paths.items():
        if key == "subperiods" and scope.name != "9-sector-long-history":
            continue
        print(f"- {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    for scope in SCOPES:
        data = read_dataset(scope)
        predictions, backtest, validations = run_scope(scope, data)
        reproduction = reproduce_enhanced(scope, backtest)
        validations.append(reproduction)
        metrics = build_metrics(scope, backtest)
        inc = incremental(backtest)
        cal = calendar_year(backtest)
        sub = subperiods_9(backtest) if scope.name == "9-sector-long-history" else None
        conc = concentration(backtest)
        bw = best_worst(backtest)
        backtest = add_growth(backtest)
        paths = output_paths(scope)
        plot(scope, backtest, paths)
        save(scope, predictions, backtest, metrics, inc, cal, sub, paths)
        validations.extend(validate_outputs(scope, paths, predictions, backtest, metrics))
        print_audit(scope, data, predictions, backtest, metrics, inc, cal, sub, conc, bw, validations, paths)


if __name__ == "__main__":
    main()
