"""Run enhanced walk-forward Ridge sector-ranking models."""

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


@dataclass(frozen=True)
class ScopeConfig:
    name: str
    input_path: Path
    baseline_backtest_path: Path
    sectors: list[str]
    min_train_months: int
    prediction_output: Path
    backtest_output: Path
    metrics_output: Path
    coefficients_output: Path
    comparison_output: Path
    plot_output: Path


SCOPES = [
    ScopeConfig(
        "11-sector",
        DATA_DIR / "enhanced_model_dataset_11_sector.csv",
        DATA_DIR / "baseline_backtest_11_sector.csv",
        SECTORS_11,
        48,
        DATA_DIR / "enhanced_predictions_11_sector.csv",
        DATA_DIR / "enhanced_backtest_11_sector.csv",
        DATA_DIR / "enhanced_metrics_11_sector.csv",
        DATA_DIR / "enhanced_final_coefficients_11_sector.csv",
        DATA_DIR / "enhanced_vs_baseline_comparison_11_sector.csv",
        OUTPUT_DIR / "enhanced_cumulative_returns_11_sector.png",
    ),
    ScopeConfig(
        "9-sector-long-history",
        DATA_DIR / "enhanced_model_dataset_9_sector_long_history.csv",
        DATA_DIR / "baseline_backtest_9_sector_long_history.csv",
        SECTORS_9,
        120,
        DATA_DIR / "enhanced_predictions_9_sector_long_history.csv",
        DATA_DIR / "enhanced_backtest_9_sector_long_history.csv",
        DATA_DIR / "enhanced_metrics_9_sector_long_history.csv",
        DATA_DIR / "enhanced_final_coefficients_9_sector_long_history.csv",
        DATA_DIR / "enhanced_vs_baseline_comparison_9_sector_long_history.csv",
        OUTPUT_DIR / "enhanced_cumulative_returns_9_sector_long_history.png",
    ),
]


def sector_features(ticker: str) -> list[str]:
    return [
        f"sector_momentum_3m_{ticker}",
        f"sector_momentum_6m_{ticker}",
        f"sector_relative_momentum_6m_{ticker}",
        f"sector_volatility_6m_{ticker}",
    ]


def model_features(ticker: str) -> list[str]:
    return [*MACRO_FEATURES, *sector_features(ticker)]


def target_return_col(ticker: str) -> str:
    return f"target_next_return_{ticker}"


def target_rank_col(ticker: str) -> str:
    return f"target_next_rank_{ticker}"


def predicted_return_col(ticker: str) -> str:
    return f"predicted_return_{ticker}"


def predicted_rank_col(ticker: str) -> str:
    return f"predicted_rank_{ticker}"


def actual_return_col(ticker: str) -> str:
    return f"actual_next_return_{ticker}"


def actual_rank_col(ticker: str) -> str:
    return f"actual_next_rank_{ticker}"


def make_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


def read_enhanced_dataset(config: ScopeConfig) -> pd.DataFrame:
    required = [
        *MACRO_FEATURES,
        *[feature for ticker in config.sectors for feature in sector_features(ticker)],
        *[target_return_col(ticker) for ticker in config.sectors],
        *[target_rank_col(ticker) for ticker in config.sectors],
        "target_best_sector",
    ]
    data = pd.read_csv(config.input_path, parse_dates=["date"]).set_index("date").sort_index()
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"{config.input_path.name} missing columns: {', '.join(missing)}")
    if data.index.has_duplicates:
        raise ValueError(f"{config.input_path.name} contains duplicate dates.")
    if data.isna().any().any():
        raise ValueError(f"{config.input_path.name} contains missing values.")
    if "date" in required:
        raise ValueError("Date metadata was included in model feature requirements.")
    return data.loc[:, required]


def validate_sector_feature_isolation(config: ScopeConfig) -> list[str]:
    for ticker in config.sectors:
        features = model_features(ticker)
        if len(features) != 13:
            raise ValueError(f"{config.name} {ticker} does not have exactly 13 features.")
        other_sector_features = [
            f for other in config.sectors if other != ticker for f in sector_features(other)
        ]
        if set(features) & set(other_sector_features):
            raise ValueError(f"{config.name} {ticker} includes another sector's feature.")
    return [
        f"{config.name}: date metadata excluded from model features",
        f"{config.name}: each sector model uses exactly 13 features",
        f"{config.name}: each sector model uses only its own 4 sector-specific features",
    ]


def run_walk_forward(config: ScopeConfig, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    validations = validate_sector_feature_isolation(config)
    rows = []
    if len(data) <= config.min_train_months:
        raise ValueError(f"{config.name} has too few rows for training window.")

    for row_number in range(config.min_train_months, len(data)):
        feature_date = data.index[row_number]
        train = data.iloc[:row_number]
        if not train.index.max() < feature_date:
            raise ValueError(f"{config.name} leakage detected at {feature_date.date()}.")

        predicted = {}
        actual = {
            ticker: float(data.at[feature_date, target_return_col(ticker)])
            for ticker in config.sectors
        }
        row = {
            "feature_date": feature_date,
            "predicted_return_month": feature_date + pd.offsets.MonthEnd(1),
            "training_start_date": train.index.min(),
            "training_end_date": train.index.max(),
            "training_observations": len(train),
        }
        for ticker in config.sectors:
            features = model_features(ticker)
            model = make_pipeline()
            model.fit(train.loc[:, features], train[target_return_col(ticker)])
            if int(model.named_steps["scaler"].n_samples_seen_) != len(train):
                raise ValueError(f"{config.name} scaler validation failed for {ticker}.")
            predicted[ticker] = float(model.predict(data.loc[[feature_date], features])[0])

        predicted_rank = pd.Series(predicted).rank(ascending=False, method="min")
        actual_rank = pd.Series(actual).rank(ascending=False, method="min")
        predicted_best = max(predicted, key=predicted.get)
        actual_best = max(actual, key=actual.get)
        top3 = sorted(predicted, key=predicted.get, reverse=True)[:3]

        for ticker in config.sectors:
            row[predicted_return_col(ticker)] = predicted[ticker]
            row[predicted_rank_col(ticker)] = int(predicted_rank[ticker])
            row[actual_return_col(ticker)] = actual[ticker]
            row[actual_rank_col(ticker)] = int(actual_rank[ticker])
        row["predicted_best_sector"] = predicted_best
        row["actual_best_sector"] = actual_best
        row["selected_top3_sectors"] = ",".join(top3)
        row["top3_selected_sectors"] = ",".join(top3)
        row["top1_strategy_return"] = actual[predicted_best]
        row["top3_strategy_return"] = sum(actual[ticker] for ticker in top3) / 3
        row["equal_weight_benchmark_return"] = sum(actual.values()) / len(actual)
        row["rank_ic"] = pd.Series(predicted).corr(pd.Series(actual), method="spearman")
        rows.append(row)

    predictions = pd.DataFrame(rows).set_index("feature_date").sort_index()
    validations.extend(validate_predictions(config, predictions))
    validations.extend(
        [
            f"{config.name}: no random split used",
            f"{config.name}: scaler is fit inside each walk-forward loop using training data only",
            f"{config.name}: final coefficients are not used for backtest predictions",
        ]
    )
    return predictions, validations


def validate_predictions(config: ScopeConfig, predictions: pd.DataFrame) -> list[str]:
    if predictions.empty:
        raise ValueError(f"{config.name} prediction table is empty.")
    if predictions.index.has_duplicates or not predictions.index.is_monotonic_increasing:
        raise ValueError(f"{config.name} prediction dates are not sorted and unique.")
    expected_first_offset = config.min_train_months
    if predictions.iloc[0]["training_observations"] != expected_first_offset:
        raise ValueError(f"{config.name} first prediction violates training window.")
    for date, row in predictions.iterrows():
        if not row["training_end_date"] < date:
            raise ValueError(f"{config.name} max training date is not before prediction date.")
        if sum(row[predicted_rank_col(t)] == 1 for t in config.sectors) != 1:
            raise ValueError(f"{config.name} rank 1 validation failed.")
        top3 = str(row["selected_top3_sectors"]).split(",")
        top3_return = sum(row[actual_return_col(t)] for t in top3) / 3
        equal_weight = sum(row[actual_return_col(t)] for t in config.sectors) / len(config.sectors)
        if abs(top3_return - row["top3_strategy_return"]) > 1e-12:
            raise ValueError(f"{config.name} top3 return mismatch.")
        if abs(equal_weight - row["equal_weight_benchmark_return"]) > 1e-12:
            raise ValueError(f"{config.name} equal-weight return mismatch.")
    return [
        f"{config.name}: prediction table is non-empty",
        f"{config.name}: prediction dates are sorted and unique",
        f"{config.name}: all predicted rankings contain exactly one rank 1",
        f"{config.name}: top3 return equals selected-sector average actual return",
        f"{config.name}: equal-weight return equals all-sector average actual return",
        f"{config.name}: max training date < current prediction feature date",
    ]


def add_growth_columns(backtest: pd.DataFrame) -> pd.DataFrame:
    result = backtest.copy()
    for col in ["top1_strategy_return", "top3_strategy_return", "equal_weight_benchmark_return"]:
        growth = (1 + result[col]).cumprod()
        result[f"{col}_growth_of_1_start"] = growth.shift(1).fillna(1.0)
        result[f"{col}_growth_of_1"] = growth
    return result


def build_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "predicted_return_month",
        "top1_strategy_return",
        "top3_strategy_return",
        "equal_weight_benchmark_return",
        "rank_ic",
        "predicted_best_sector",
        "actual_best_sector",
        "selected_top3_sectors",
        "top3_selected_sectors",
    ]
    return add_growth_columns(predictions.loc[:, cols])


def max_drawdown(returns: pd.Series) -> float:
    growth = (1 + returns).cumprod()
    return float((growth / growth.cummax() - 1).min())


def strategy_metrics(name: str, returns: pd.Series) -> dict[str, float | str]:
    months = len(returns)
    cumulative = float((1 + returns).prod() - 1)
    ann_return = float((1 + cumulative) ** (12 / months) - 1)
    ann_vol = float(returns.std(ddof=1) * (12**0.5))
    return {
        "metric_group": "strategy",
        "strategy": name,
        "number_of_out_of_sample_months": months,
        "cumulative_return": cumulative,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio_rf_0": ann_return / ann_vol if ann_vol != 0 else "not_applicable",
        "maximum_drawdown": max_drawdown(returns),
        "positive_month_rate": float((returns > 0).mean()),
    }


def ranking_diagnostics(config: ScopeConfig, predictions: pd.DataFrame) -> dict[str, float | str]:
    excess = predictions["top3_strategy_return"] - predictions["equal_weight_benchmark_return"]
    top3_sets = predictions["selected_top3_sectors"].str.split(",")
    turnovers = []
    previous = None
    for selected in top3_sets:
        current = set(selected)
        if previous is not None:
            turnovers.append(1 - len(current & previous) / 3)
        previous = current
    tracking_error = float(excess.std(ddof=1) * (12**0.5))
    return {
        "metric_group": "ranking",
        "strategy": "ranking_diagnostics",
        "top1_accuracy": float((predictions["predicted_best_sector"] == predictions["actual_best_sector"]).mean()),
        "random_choice_top1_benchmark": 1 / len(config.sectors),
        "average_monthly_rank_ic": float(predictions["rank_ic"].mean()),
        "median_monthly_rank_ic": float(predictions["rank_ic"].median()),
        "positive_rank_ic_rate": float((predictions["rank_ic"] > 0).mean()),
        "average_monthly_top3_excess_return": float(excess.mean()),
        "annualized_tracking_error_top3_vs_benchmark": tracking_error,
        "information_ratio_top3_vs_benchmark": float(excess.mean() * 12 / tracking_error) if tracking_error != 0 else "not_applicable",
        "average_monthly_top3_turnover": float(pd.Series(turnovers).mean()) if turnovers else "not_applicable",
    }


def build_metrics(config: ScopeConfig, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        strategy_metrics("top1", predictions["top1_strategy_return"]),
        strategy_metrics("top3", predictions["top3_strategy_return"]),
        strategy_metrics("equal_weight_benchmark", predictions["equal_weight_benchmark_return"]),
        ranking_diagnostics(config, predictions),
    ]
    return pd.DataFrame(rows).fillna("not_applicable")


def comparison_metrics(name: str, returns: pd.Series, benchmark: pd.Series) -> dict[str, float | str]:
    row = strategy_metrics(name, returns)
    row["average_monthly_excess_return_vs_equal_weight"] = float((returns - benchmark).mean())
    te = float((returns - benchmark).std(ddof=1) * (12**0.5))
    row["information_ratio_vs_equal_weight"] = float((returns - benchmark).mean() * 12 / te) if te != 0 else "not_applicable"
    return row


def load_matched_baseline(config: ScopeConfig, enhanced_backtest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(
        config.baseline_backtest_path,
        parse_dates=["feature_date", "predicted_return_month"],
    ).set_index("feature_date").sort_index()
    matched = baseline.loc[enhanced_backtest.index].copy()
    if not matched.index.equals(enhanced_backtest.index):
        raise ValueError(f"{config.name} matched baseline dates do not align.")
    if not matched["predicted_return_month"].equals(enhanced_backtest["predicted_return_month"]):
        raise ValueError(f"{config.name} predicted return months do not align.")
    if (matched["equal_weight_benchmark_return"] - enhanced_backtest["equal_weight_benchmark_return"]).abs().max() > 1e-12:
        raise ValueError(f"{config.name} matched benchmark returns do not reconcile.")

    comparison = pd.DataFrame(
        [
            comparison_metrics("enhanced_top3", enhanced_backtest["top3_strategy_return"], enhanced_backtest["equal_weight_benchmark_return"]),
            comparison_metrics("matched_baseline_top3", matched["top3_strategy_return"], matched["equal_weight_benchmark_return"]),
            comparison_metrics("equal_weight_benchmark", enhanced_backtest["equal_weight_benchmark_return"], enhanced_backtest["equal_weight_benchmark_return"]),
        ]
    ).fillna("not_applicable")
    impact = incremental_impact(config, enhanced_backtest, matched)
    return matched, comparison, impact


def incremental_impact(config: ScopeConfig, enhanced: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    enhanced_metrics = strategy_metrics("enhanced_top3", enhanced["top3_strategy_return"])
    baseline_metrics = strategy_metrics("baseline_top3", baseline["top3_strategy_return"])
    diff = enhanced["top3_strategy_return"] - baseline["top3_strategy_return"]
    return pd.DataFrame(
        [
            {
                "scope": config.name,
                "enhanced_minus_baseline_cumulative_return": enhanced_metrics["cumulative_return"] - baseline_metrics["cumulative_return"],
                "enhanced_minus_baseline_annualized_return": enhanced_metrics["annualized_return"] - baseline_metrics["annualized_return"],
                "enhanced_minus_baseline_sharpe_ratio": enhanced_metrics["sharpe_ratio_rf_0"] - baseline_metrics["sharpe_ratio_rf_0"],
                "enhanced_minus_baseline_maximum_drawdown": enhanced_metrics["maximum_drawdown"] - baseline_metrics["maximum_drawdown"],
                "enhanced_minus_baseline_average_monthly_return": float(diff.mean()),
                "months_enhanced_top3_outperformed_baseline_top3": int((diff > 0).sum()),
                "months_enhanced_top3_underperformed_baseline_top3": int((diff < 0).sum()),
            }
        ]
    )


def best_worst_enhancement_months(enhanced: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.DataFrame(
        {
            "feature_date": enhanced.index,
            "predicted_return_month": enhanced["predicted_return_month"].values,
            "enhanced_selected_top3_sectors": enhanced["selected_top3_sectors"].values,
            "baseline_selected_top3_sectors": baseline["top3_selected_sectors"].values,
            "enhanced_top3_return": enhanced["top3_strategy_return"].values,
            "baseline_top3_return": baseline["top3_strategy_return"].values,
            "equal_weight_benchmark_return": enhanced["equal_weight_benchmark_return"].values,
        }
    )
    table["enhanced_minus_baseline_return_difference"] = (
        table["enhanced_top3_return"] - table["baseline_top3_return"]
    )
    return (
        table.nlargest(5, "enhanced_minus_baseline_return_difference"),
        table.nsmallest(5, "enhanced_minus_baseline_return_difference"),
    )


def fit_final_coefficients(config: ScopeConfig, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in config.sectors:
        features = model_features(ticker)
        model = make_pipeline()
        model.fit(data.loc[:, features], data[target_return_col(ticker)])
        for feature, coef in zip(features, model.named_steps["ridge"].coef_, strict=True):
            rows.append(
                {
                    "coefficient_type": "final_full_sample_explanatory_standardized",
                    "sector": ticker,
                    "feature": feature,
                    "coefficient": coef,
                    "ridge_alpha": RIDGE_ALPHA,
                    "training_start_date": data.index.min().date().isoformat(),
                    "training_end_date": data.index.max().date().isoformat(),
                    "training_observations": len(data),
                    "used_for_backtest_predictions": False,
                }
            )
    return pd.DataFrame(rows)


def plot_matched_returns(config: ScopeConfig, enhanced: pd.DataFrame, baseline: pd.DataFrame) -> None:
    start_date = enhanced["predicted_return_month"].min() - pd.offsets.MonthEnd(1)
    dates = pd.concat([pd.Series([start_date]), enhanced["predicted_return_month"].reset_index(drop=True)], ignore_index=True)

    def growth(returns: pd.Series) -> pd.Series:
        return pd.concat([pd.Series([1.0]), (1 + returns).cumprod().reset_index(drop=True)], ignore_index=True)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, growth(enhanced["top3_strategy_return"]), label="Enhanced top3")
    plt.plot(dates, growth(baseline["top3_strategy_return"]), label="Macro-only baseline top3")
    plt.plot(dates, growth(enhanced["equal_weight_benchmark_return"]), label="Equal-weight benchmark")
    plt.title(f"Enhanced vs Baseline Matched Returns: {config.name}")
    plt.xlabel("Predicted return month")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    config.plot_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.plot_output, dpi=150)
    plt.close()


def save_outputs(config: ScopeConfig, predictions: pd.DataFrame, backtest: pd.DataFrame, metrics: pd.DataFrame, coefficients: pd.DataFrame, comparison: pd.DataFrame) -> None:
    config.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.prediction_output, index_label="feature_date")
    backtest.to_csv(config.backtest_output, index_label="feature_date")
    metrics.to_csv(config.metrics_output, index=False)
    coefficients.to_csv(config.coefficients_output, index=False)
    comparison.to_csv(config.comparison_output, index=False)


def validate_outputs(config: ScopeConfig, predictions: pd.DataFrame, backtest: pd.DataFrame, metrics: pd.DataFrame, coefficients: pd.DataFrame, matched: pd.DataFrame) -> list[str]:
    if coefficients["used_for_backtest_predictions"].any():
        raise ValueError("Final coefficients were used for backtest predictions.")
    for path in [config.prediction_output, config.backtest_output, config.metrics_output, config.coefficients_output, config.comparison_output, config.plot_output]:
        if not path.exists():
            raise ValueError(f"Expected output file not found: {path}")
    if backtest["top3_strategy_return_growth_of_1_start"].iloc[0] != 1.0:
        raise ValueError("Cumulative return series does not start at 1.0.")
    if predictions.isna().any().any() or backtest.isna().any().any() or metrics.isna().any().any():
        raise ValueError(f"{config.name} output contains missing values.")
    return [
        f"{config.name}: matched-window enhanced, baseline, and benchmark dates align exactly",
        f"{config.name}: matched-window benchmark returns reconcile exactly",
        f"{config.name}: cumulative-return plot begins at 1.0",
        f"{config.name}: final coefficients are not used for backtest predictions",
        f"{config.name}: output CSV and PNG files exist",
    ]


def print_audit(config: ScopeConfig, data: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, comparison: pd.DataFrame, impact: pd.DataFrame, best: pd.DataFrame, worst: pd.DataFrame, coefficients: pd.DataFrame, validations: list[str]) -> None:
    print()
    print(f"Enhanced Model Audit: {config.name}")
    print("=" * (22 + len(config.name)))
    print("Model type: expanding-window one-model-per-sector Ridge Regression")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print("Feature specification: 9 macro features plus each sector model's own 4 trailing sector features")
    print("Date metadata is excluded from all model input feature matrices.")
    print(f"Sector universe: {', '.join(config.sectors)}")
    print(f"Minimum training window: {config.min_train_months} months")
    print(f"Input dataset date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"OOS feature dates: {predictions.index.min().date()} to {predictions.index.max().date()}")
    print(f"Predicted return months: {predictions['predicted_return_month'].min().date()} to {predictions['predicted_return_month'].max().date()}")
    print(f"Number of OOS months: {len(predictions)}")
    print()
    print("Enhanced metrics table:")
    print(metrics.to_string(index=False))
    print()
    print("Matched-window baseline comparison table:")
    print(comparison.to_string(index=False))
    print()
    print("Incremental enhancement impact:")
    print(impact.to_string(index=False))
    print()
    print("First five monthly prediction summaries:")
    print(predictions[["predicted_return_month", "predicted_best_sector", "actual_best_sector", "selected_top3_sectors", "top3_strategy_return", "equal_weight_benchmark_return", "rank_ic"]].head().to_string())
    print()
    print("Best enhancement months:")
    print(best.to_string(index=False))
    print()
    print("Worst enhancement months:")
    print(worst.to_string(index=False))
    if config.name == "11-sector":
        print()
        print("11-sector five largest absolute final full-sample coefficients per sector:")
        table = coefficients.assign(abs_coef=coefficients["coefficient"].abs(), sign=coefficients["coefficient"].map(lambda x: "positive" if x >= 0 else "negative")).sort_values(["sector", "abs_coef"], ascending=[True, False]).groupby("sector").head(5)
        print(table[["sector", "feature", "coefficient", "sign"]].to_string(index=False))
    print()
    print("Output files:")
    for path in [config.prediction_output, config.backtest_output, config.metrics_output, config.coefficients_output, config.comparison_output, config.plot_output]:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    print()
    print("Validation results:")
    for validation in validations:
        print(f"- {validation}")


def main() -> None:
    for config in SCOPES:
        data = read_enhanced_dataset(config)
        predictions, validations = run_walk_forward(config, data)
        backtest = build_backtest(predictions)
        metrics = build_metrics(config, predictions)
        matched, comparison, impact = load_matched_baseline(config, backtest)
        best, worst = best_worst_enhancement_months(backtest, matched)
        coefficients = fit_final_coefficients(config, data)
        plot_matched_returns(config, backtest, matched)
        save_outputs(config, predictions, backtest, metrics, coefficients, comparison)
        validations.extend(validate_outputs(config, predictions, backtest, metrics, coefficients, matched))
        print_audit(config, data, predictions, metrics, comparison, impact, best, worst, coefficients, validations)

    print()
    print("Global validation summary:")
    print("- No random train-test split was used.")
    print("- Baseline script and baseline outputs were not modified.")
    print("- Enhanced full-sample coefficients were fit only after walk-forward evaluation.")
    print(f"- Completed {len(SCOPES)} enhanced dataset scopes.")


if __name__ == "__main__":
    main()
