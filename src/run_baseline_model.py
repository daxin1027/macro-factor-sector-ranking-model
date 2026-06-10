"""Run walk-forward Ridge baseline sector-ranking models."""

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

FEATURE_COLUMNS = [
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
RIDGE_ALPHA = 10.0


@dataclass(frozen=True)
class ScopeConfig:
    name: str
    input_path: Path
    sectors: list[str]
    min_train_months: int
    prediction_output: Path
    backtest_output: Path
    metrics_output: Path
    coefficients_output: Path
    plot_output: Path


SCOPES = [
    ScopeConfig(
        name="11-sector",
        input_path=DATA_DIR / "model_dataset_11_sector.csv",
        sectors=SECTORS_11,
        min_train_months=48,
        prediction_output=DATA_DIR / "baseline_predictions_11_sector.csv",
        backtest_output=DATA_DIR / "baseline_backtest_11_sector.csv",
        metrics_output=DATA_DIR / "baseline_metrics_11_sector.csv",
        coefficients_output=DATA_DIR / "baseline_final_coefficients_11_sector.csv",
        plot_output=OUTPUT_DIR / "baseline_cumulative_returns_11_sector.png",
    ),
    ScopeConfig(
        name="9-sector-long-history",
        input_path=DATA_DIR / "model_dataset_9_sector_long_history.csv",
        sectors=SECTORS_9,
        min_train_months=120,
        prediction_output=DATA_DIR / "baseline_predictions_9_sector_long_history.csv",
        backtest_output=DATA_DIR / "baseline_backtest_9_sector_long_history.csv",
        metrics_output=DATA_DIR / "baseline_metrics_9_sector_long_history.csv",
        coefficients_output=DATA_DIR / "baseline_final_coefficients_9_sector_long_history.csv",
        plot_output=OUTPUT_DIR / "baseline_cumulative_returns_9_sector_long_history.png",
    ),
]


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
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def read_model_dataset(config: ScopeConfig) -> pd.DataFrame:
    required = [
        *FEATURE_COLUMNS,
        *[target_return_col(ticker) for ticker in config.sectors],
        *[target_rank_col(ticker) for ticker in config.sectors],
        "target_best_sector",
    ]
    if not config.input_path.exists():
        raise FileNotFoundError(f"Missing model-ready input file: {config.input_path}")

    data = pd.read_csv(config.input_path, parse_dates=["date"]).set_index("date")
    data = data.sort_index()

    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"{config.input_path.name} is missing columns: {', '.join(missing)}")

    if data.index.has_duplicates:
        raise ValueError(f"{config.input_path.name} contains duplicate dates.")

    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{config.input_path.name} dates are not sorted.")

    if data.loc[:, required].isna().any().any():
        raise ValueError(f"{config.input_path.name} contains missing values.")

    return data.loc[:, required]


def run_walk_forward(config: ScopeConfig, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    validations = []
    rows = []
    features = data.loc[:, FEATURE_COLUMNS]

    if len(data) <= config.min_train_months:
        raise ValueError(f"{config.name} has too few rows for the requested training window.")

    for row_number in range(config.min_train_months, len(data)):
        prediction_date = data.index[row_number]
        train = data.iloc[:row_number]
        max_train_date = train.index.max()
        if not max_train_date < prediction_date:
            raise ValueError(f"Leakage detected for {config.name}: training includes prediction date.")

        x_train = train.loc[:, FEATURE_COLUMNS]
        x_current = features.loc[[prediction_date], :]

        prediction_row = {
            "feature_date": prediction_date,
            "predicted_return_month": prediction_date + pd.offsets.MonthEnd(1),
            "training_start_date": train.index.min(),
            "training_end_date": max_train_date,
            "training_observations": len(train),
        }
        predicted_returns = {}
        actual_returns = {
            ticker: data.at[prediction_date, target_return_col(ticker)]
            for ticker in config.sectors
        }

        for ticker in config.sectors:
            model = make_pipeline()
            model.fit(x_train, train[target_return_col(ticker)])
            scaler_seen_rows = int(model.named_steps["scaler"].n_samples_seen_)
            if scaler_seen_rows != len(train):
                raise ValueError(f"Scaler fit validation failed for {config.name} {ticker}.")
            predicted_returns[ticker] = float(model.predict(x_current)[0])

        predicted_rank = pd.Series(predicted_returns).rank(
            ascending=False, method="min"
        )
        actual_rank = pd.Series(actual_returns).rank(ascending=False, method="min")
        predicted_best = max(predicted_returns, key=predicted_returns.get)
        actual_best = max(actual_returns, key=actual_returns.get)
        top3 = sorted(predicted_returns, key=predicted_returns.get, reverse=True)[:3]

        for ticker in config.sectors:
            prediction_row[predicted_return_col(ticker)] = predicted_returns[ticker]
            prediction_row[predicted_rank_col(ticker)] = int(predicted_rank[ticker])
            prediction_row[actual_return_col(ticker)] = actual_returns[ticker]
            prediction_row[actual_rank_col(ticker)] = int(actual_rank[ticker])

        prediction_row["predicted_best_sector"] = predicted_best
        prediction_row["actual_best_sector"] = actual_best
        prediction_row["top3_selected_sectors"] = ",".join(top3)
        prediction_row["top1_strategy_return"] = actual_returns[predicted_best]
        prediction_row["top3_strategy_return"] = sum(actual_returns[t] for t in top3) / 3
        prediction_row["equal_weight_benchmark_return"] = (
            sum(actual_returns.values()) / len(actual_returns)
        )
        prediction_row["rank_ic"] = pd.Series(predicted_returns).corr(
            pd.Series(actual_returns), method="spearman"
        )
        rows.append(prediction_row)

    predictions = pd.DataFrame(rows).set_index("feature_date").sort_index()
    if predictions.empty:
        raise ValueError(f"{config.name} generated an empty prediction table.")

    validations.extend(validate_predictions(config, data, predictions))
    validations.append("No random train-test split used; expanding walk-forward loop only.")
    validations.append("Scaler is fit inside each walk-forward sector model using training rows only.")
    validations.append("Final full-sample coefficients are generated after backtest and not used for predictions.")
    return predictions, validations


def validate_predictions(
    config: ScopeConfig,
    data: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[str]:
    if predictions.index.has_duplicates:
        raise ValueError(f"{config.name} prediction dates are duplicated.")
    if not predictions.index.is_monotonic_increasing:
        raise ValueError(f"{config.name} prediction dates are not sorted.")
    expected_first = data.index[config.min_train_months]
    if predictions.index.min() != expected_first:
        raise ValueError(f"{config.name} first prediction violates minimum training window.")

    for date, row in predictions.iterrows():
        if not row["training_end_date"] < date:
            raise ValueError(f"{config.name} leakage detected at {date.date()}.")

        pred_rank_ones = sum(row[predicted_rank_col(ticker)] == 1 for ticker in config.sectors)
        if pred_rank_ones != 1:
            raise ValueError(f"{config.name} does not have exactly one predicted rank 1.")

        top3 = str(row["top3_selected_sectors"]).split(",")
        top3_return = sum(row[actual_return_col(ticker)] for ticker in top3) / 3
        if abs(top3_return - row["top3_strategy_return"]) > 1e-12:
            raise ValueError(f"{config.name} top3 strategy return mismatch.")

        equal_weight = sum(row[actual_return_col(ticker)] for ticker in config.sectors) / len(config.sectors)
        if abs(equal_weight - row["equal_weight_benchmark_return"]) > 1e-12:
            raise ValueError(f"{config.name} equal-weight benchmark mismatch.")

    return [
        f"{config.name}: prediction table is non-empty",
        f"{config.name}: prediction dates are sorted and unique",
        f"{config.name}: first prediction occurs after {config.min_train_months} training months",
        f"{config.name}: every prediction month has exactly one predicted rank 1",
        f"{config.name}: top3 returns equal selected-sector average actual returns",
        f"{config.name}: equal-weight returns equal full-universe average actual returns",
        f"{config.name}: max training date is strictly before each prediction date",
    ]


def build_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    backtest = predictions[
        [
            "predicted_return_month",
            "top1_strategy_return",
            "top3_strategy_return",
            "equal_weight_benchmark_return",
            "rank_ic",
            "predicted_best_sector",
            "actual_best_sector",
            "top3_selected_sectors",
        ]
    ].copy()
    for column in [
        "top1_strategy_return",
        "top3_strategy_return",
        "equal_weight_benchmark_return",
    ]:
        growth_end = (1 + backtest[column]).cumprod()
        backtest[f"{column}_growth_of_1_start"] = growth_end.shift(1).fillna(1.0)
        backtest[f"{column}_growth_of_1"] = growth_end
    return backtest


def max_drawdown(returns: pd.Series) -> float:
    growth = (1 + returns).cumprod()
    drawdown = growth / growth.cummax() - 1
    return float(drawdown.min())


def strategy_metrics(name: str, returns: pd.Series) -> dict[str, float | str]:
    months = len(returns)
    cumulative = float((1 + returns).prod() - 1)
    annualized_return = float((1 + cumulative) ** (12 / months) - 1)
    annualized_vol = float(returns.std(ddof=1) * (12**0.5))
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else pd.NA
    return {
        "metric_group": "strategy",
        "strategy": name,
        "number_of_out_of_sample_months": months,
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio_rf_0": sharpe,
        "maximum_drawdown": max_drawdown(returns),
        "positive_month_rate": float((returns > 0).mean()),
    }


def ranking_diagnostics(predictions: pd.DataFrame) -> dict[str, float | str]:
    top3_excess = (
        predictions["top3_strategy_return"]
        - predictions["equal_weight_benchmark_return"]
    )
    top3_sets = predictions["top3_selected_sectors"].str.split(",")
    turnovers = []
    previous = None
    for selected in top3_sets:
        current = set(selected)
        if previous is not None:
            turnovers.append(1 - len(current & previous) / 3)
        previous = current
    tracking_error = float(top3_excess.std(ddof=1) * (12**0.5))
    info_ratio = (
        float(top3_excess.mean() * 12 / tracking_error)
        if tracking_error != 0
        else pd.NA
    )
    return {
        "metric_group": "ranking",
        "strategy": "ranking_diagnostics",
        "top1_accuracy": float(
            (predictions["predicted_best_sector"] == predictions["actual_best_sector"]).mean()
        ),
        "average_monthly_rank_ic": float(predictions["rank_ic"].mean()),
        "median_monthly_rank_ic": float(predictions["rank_ic"].median()),
        "positive_rank_ic_rate": float((predictions["rank_ic"] > 0).mean()),
        "average_monthly_top3_excess_return": float(top3_excess.mean()),
        "annualized_tracking_error_top3_vs_benchmark": tracking_error,
        "information_ratio_top3_vs_benchmark": info_ratio,
        "average_monthly_top3_turnover": float(pd.Series(turnovers).mean()) if turnovers else pd.NA,
    }


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        strategy_metrics("top1", predictions["top1_strategy_return"]),
        strategy_metrics("top3", predictions["top3_strategy_return"]),
        strategy_metrics("equal_weight_benchmark", predictions["equal_weight_benchmark_return"]),
        ranking_diagnostics(predictions),
    ]
    metrics = pd.DataFrame(rows)
    return metrics.fillna("not_applicable")


def fit_final_coefficients(config: ScopeConfig, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = data.loc[:, FEATURE_COLUMNS]
    for ticker in config.sectors:
        model = make_pipeline()
        model.fit(x, data[target_return_col(ticker)])
        coefficients = model.named_steps["ridge"].coef_
        for feature, coefficient in zip(FEATURE_COLUMNS, coefficients, strict=True):
            rows.append(
                {
                    "coefficient_type": "final_full_sample_explanatory_standardized",
                    "sector": ticker,
                    "feature": feature,
                    "coefficient": coefficient,
                    "ridge_alpha": RIDGE_ALPHA,
                    "training_start_date": data.index.min().date().isoformat(),
                    "training_end_date": data.index.max().date().isoformat(),
                    "training_observations": len(data),
                    "used_for_backtest_predictions": False,
                }
            )
    return pd.DataFrame(rows)


def plot_cumulative_returns(config: ScopeConfig, backtest: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    start_date = backtest["predicted_return_month"].min() - pd.offsets.MonthEnd(1)
    plot_dates = pd.concat(
        [
            pd.Series([start_date]),
            backtest["predicted_return_month"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    plt.plot(
        plot_dates,
        pd.concat(
            [
                pd.Series([1.0]),
                backtest["top1_strategy_return_growth_of_1"].reset_index(drop=True),
            ],
            ignore_index=True,
        ),
        label="Top 1 strategy",
    )
    plt.plot(
        plot_dates,
        pd.concat(
            [
                pd.Series([1.0]),
                backtest["top3_strategy_return_growth_of_1"].reset_index(drop=True),
            ],
            ignore_index=True,
        ),
        label="Top 3 strategy",
    )
    plt.plot(
        plot_dates,
        pd.concat(
            [
                pd.Series([1.0]),
                backtest["equal_weight_benchmark_return_growth_of_1"].reset_index(drop=True),
            ],
            ignore_index=True,
        ),
        label="Equal-weight benchmark",
    )
    plt.title(f"Baseline Ridge Sector Rotation: {config.name}")
    plt.xlabel("Predicted return month")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    config.plot_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.plot_output, dpi=150)
    plt.close()


def save_outputs(
    config: ScopeConfig,
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    config.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.prediction_output, index_label="feature_date")
    backtest.to_csv(config.backtest_output, index_label="feature_date")
    metrics.to_csv(config.metrics_output, index=False)
    coefficients.to_csv(config.coefficients_output, index=False)
    plot_cumulative_returns(config, backtest)


def print_scope_audit(
    config: ScopeConfig,
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    validations: list[str],
) -> None:
    print()
    print(f"Baseline Audit: {config.name}")
    print("=" * (16 + len(config.name)))
    print("Model type: expanding-window one-model-per-sector Ridge Regression")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"Sector universe: {', '.join(config.sectors)}")
    print(f"Minimum training window: {config.min_train_months} months")
    print(f"Input dataset date range: {data.index.min().date()} to {data.index.max().date()}")
    print(
        "Out-of-sample feature dates: "
        f"{predictions.index.min().date()} to {predictions.index.max().date()}"
    )
    print(
        "Predicted return months: "
        f"{predictions['predicted_return_month'].min().date()} "
        f"to {predictions['predicted_return_month'].max().date()}"
    )
    print(f"Number of out-of-sample months: {len(predictions)}")
    print()
    print("Metrics table:")
    print(metrics.to_string(index=False))
    print()
    print("First five monthly prediction summaries:")
    columns = [
        "predicted_return_month",
        "predicted_best_sector",
        "actual_best_sector",
        "top1_strategy_return",
        "top3_strategy_return",
        "equal_weight_benchmark_return",
        "rank_ic",
    ]
    print(predictions.loc[:, columns].head().to_string())
    print()
    print("Output files:")
    for path in [
        config.prediction_output,
        config.backtest_output,
        config.metrics_output,
        config.coefficients_output,
        config.plot_output,
    ]:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    print()
    print("Validation results:")
    for validation in validations:
        print(f"- {validation}")


def main() -> None:
    all_validations = []
    for config in SCOPES:
        data = read_model_dataset(config)
        predictions, validations = run_walk_forward(config, data)
        backtest = build_backtest(predictions)
        metrics = build_metrics(predictions)
        coefficients = fit_final_coefficients(config, data)

        if backtest.empty or metrics.empty or coefficients.empty:
            raise ValueError(f"{config.name} generated an empty output.")
        if coefficients["used_for_backtest_predictions"].any():
            raise ValueError("Final coefficients were marked as used for backtest predictions.")

        save_outputs(config, predictions, backtest, metrics, coefficients)
        all_validations.extend(validations)
        print_scope_audit(config, data, predictions, metrics, validations)

    print()
    print("Global validation summary:")
    print("- No random train-test split was used.")
    print("- Final full-sample coefficients were fit only after walk-forward predictions.")
    print(f"- Completed {len(SCOPES)} dataset scopes.")
    print(f"- Total validation checks reported: {len(all_validations)}")


if __name__ == "__main__":
    main()
