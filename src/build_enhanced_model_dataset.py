"""Build enhanced model-ready datasets with sector momentum and volatility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SECTOR_RETURNS_INPUT = DATA_DIR / "sector_returns_monthly.csv"
MODEL_11_INPUT = DATA_DIR / "model_dataset_11_sector.csv"
MODEL_9_INPUT = DATA_DIR / "model_dataset_9_sector_long_history.csv"
OUTPUT_11 = DATA_DIR / "enhanced_model_dataset_11_sector.csv"
OUTPUT_9 = DATA_DIR / "enhanced_model_dataset_9_sector_long_history.csv"

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
    model_input: Path
    output_path: Path
    sectors: list[str]


SCOPES = [
    ScopeConfig("11-sector", MODEL_11_INPUT, OUTPUT_11, SECTORS_11),
    ScopeConfig("9-sector-long-history", MODEL_9_INPUT, OUTPUT_9, SECTORS_9),
]


def momentum_3m_col(ticker: str) -> str:
    return f"sector_momentum_3m_{ticker}"


def momentum_6m_col(ticker: str) -> str:
    return f"sector_momentum_6m_{ticker}"


def relative_momentum_6m_col(ticker: str) -> str:
    return f"sector_relative_momentum_6m_{ticker}"


def volatility_6m_col(ticker: str) -> str:
    return f"sector_volatility_6m_{ticker}"


def target_return_col(ticker: str) -> str:
    return f"target_next_return_{ticker}"


def target_rank_col(ticker: str) -> str:
    return f"target_next_rank_{ticker}"


def enhanced_feature_columns(sectors: list[str]) -> list[str]:
    columns = []
    for ticker in sectors:
        columns.extend(
            [
                momentum_3m_col(ticker),
                momentum_6m_col(ticker),
                relative_momentum_6m_col(ticker),
                volatility_6m_col(ticker),
            ]
        )
    return columns


def read_indexed_csv(path: Path, date_column: str, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    data = pd.read_csv(path, parse_dates=[date_column])
    missing = sorted(set([date_column, *required_columns]) - set(data.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")

    data = data.loc[:, [date_column, *required_columns]].rename(
        columns={date_column: "date"}
    )
    data = data.set_index("date").sort_index()

    if data.index.has_duplicates:
        raise ValueError(f"{path.name} contains duplicate dates.")

    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} dates are not sorted.")

    return data


def validate_no_current_incomplete_month(data: pd.DataFrame, name: str) -> None:
    today = pd.Timestamp.today().normalize()
    if today < (today + pd.offsets.MonthEnd(0)).normalize():
        current_period = today.to_period("M")
        if (data.index.to_period("M") == current_period).any():
            raise ValueError(f"{name} includes the current incomplete calendar month.")


def read_sector_returns() -> pd.DataFrame:
    returns = read_indexed_csv(SECTOR_RETURNS_INPUT, "Date", SECTORS_11)
    validate_no_current_incomplete_month(returns, SECTOR_RETURNS_INPUT.name)
    return returns


def read_model_dataset(config: ScopeConfig) -> pd.DataFrame:
    required = [
        *MACRO_FEATURES,
        *[target_return_col(ticker) for ticker in config.sectors],
        *[target_rank_col(ticker) for ticker in config.sectors],
        "target_best_sector",
    ]
    data = read_indexed_csv(config.model_input, "date", required)
    if data.isna().any().any():
        raise ValueError(f"{config.model_input.name} contains missing values.")
    validate_no_current_incomplete_month(data, config.model_input.name)
    return data


def compounded_return(window: pd.Series) -> float:
    return float((1 + window).prod() - 1)


def build_trailing_features(
    sector_returns: pd.DataFrame,
    sectors: list[str],
) -> pd.DataFrame:
    """Build strict trailing features that use returns through feature month t only."""
    relevant_returns = sector_returns.loc[:, sectors].copy()
    features = pd.DataFrame(index=relevant_returns.index)

    for ticker in sectors:
        returns = relevant_returns[ticker]
        features[momentum_3m_col(ticker)] = returns.rolling(
            window=3,
            min_periods=3,
        ).apply(compounded_return, raw=False)
        features[momentum_6m_col(ticker)] = returns.rolling(
            window=6,
            min_periods=6,
        ).apply(compounded_return, raw=False)
        features[volatility_6m_col(ticker)] = (
            returns.rolling(window=6, min_periods=6).std() * (12**0.5)
        )

    momentum_6m = features[[momentum_6m_col(ticker) for ticker in sectors]]
    universe_average = momentum_6m.mean(axis=1, skipna=False)
    for ticker in sectors:
        features[relative_momentum_6m_col(ticker)] = (
            features[momentum_6m_col(ticker)] - universe_average
        )

    return features.loc[:, enhanced_feature_columns(sectors)]


def merge_enhanced_dataset(
    model_data: pd.DataFrame,
    trailing_features: pd.DataFrame,
    config: ScopeConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    enhanced_columns = enhanced_feature_columns(config.sectors)
    merged = model_data.join(trailing_features.loc[:, enhanced_columns], how="left")

    required_columns = [
        *MACRO_FEATURES,
        *enhanced_columns,
        *[target_return_col(ticker) for ticker in config.sectors],
        *[target_rank_col(ticker) for ticker in config.sectors],
        "target_best_sector",
    ]
    missing_features = merged.loc[:, enhanced_columns].isna().any(axis=1)
    missing_macro = merged.loc[:, MACRO_FEATURES].isna().any(axis=1)
    missing_targets = merged[
        [target_return_col(ticker) for ticker in config.sectors]
    ].isna().any(axis=1)
    removed = missing_features | missing_macro | missing_targets

    output = merged.loc[~removed, required_columns].copy()
    removal_summary = {
        "starting_rows": len(merged),
        "rows_removed_trailing_window_requirements": int(missing_features.sum()),
        "rows_removed_missing_macro_features": int(missing_macro.sum()),
        "rows_removed_missing_targets": int(missing_targets.sum()),
        "total_unique_rows_removed": int(removed.sum()),
        "final_rows": len(output),
    }
    if removal_summary["starting_rows"] - removal_summary["total_unique_rows_removed"] != len(output):
        raise ValueError(f"{config.name} row-removal accounting does not reconcile.")

    return output, removal_summary


def validate_targets_unchanged(
    original: pd.DataFrame,
    enhanced: pd.DataFrame,
    sectors: list[str],
    config_name: str,
) -> None:
    target_columns = [
        *[target_return_col(ticker) for ticker in sectors],
        *[target_rank_col(ticker) for ticker in sectors],
        "target_best_sector",
    ]
    original_overlap = original.loc[enhanced.index, target_columns]
    enhanced_targets = enhanced.loc[:, target_columns]
    if not original_overlap.equals(enhanced_targets):
        raise ValueError(f"{config_name} target columns changed after enhancement.")


def validate_next_month_targets(
    enhanced: pd.DataFrame,
    sector_returns: pd.DataFrame,
    sectors: list[str],
    config_name: str,
) -> None:
    for date, row in enhanced.iterrows():
        next_month = date + pd.offsets.MonthEnd(1)
        if next_month not in sector_returns.index:
            raise ValueError(f"{config_name} missing next-month return for {date.date()}.")
        for ticker in sectors:
            target_value = row[target_return_col(ticker)]
            source_value = sector_returns.at[next_month, ticker]
            if abs(target_value - source_value) > 1e-12:
                raise ValueError(
                    f"{config_name} target return mismatch for {ticker} at {date.date()}."
                )


def validate_no_future_feature_use(
    enhanced: pd.DataFrame,
    sector_returns: pd.DataFrame,
    sectors: list[str],
    config_name: str,
) -> list[str]:
    for date in enhanced.index:
        for window in [3, 6]:
            window_dates = sector_returns.loc[:date].tail(window).index
            if len(window_dates) != window or window_dates.max() > date:
                raise ValueError(
                    f"{config_name} future return leakage detected for {date.date()}."
                )
        momentum_values = [
            enhanced.at[date, momentum_6m_col(ticker)] for ticker in sectors
        ]
        average_momentum = sum(momentum_values) / len(momentum_values)
        for ticker in sectors:
            expected_relative = enhanced.at[date, momentum_6m_col(ticker)] - average_momentum
            actual_relative = enhanced.at[date, relative_momentum_6m_col(ticker)]
            if abs(expected_relative - actual_relative) > 1e-12:
                raise ValueError(
                    f"{config_name} relative momentum mismatch at {date.date()}."
                )

    return [
        f"{config_name}: 3-month momentum uses only return dates <= feature date",
        f"{config_name}: 6-month momentum uses only return dates <= feature date",
        f"{config_name}: relative momentum uses same-date trailing universe features",
        f"{config_name}: 6-month volatility uses only return dates <= feature date",
        f"{config_name}: target returns remain next-calendar-month realized returns",
    ]


def validate_relative_momentum_zero_sum(
    enhanced: pd.DataFrame,
    sectors: list[str],
    config_name: str,
) -> tuple[list[str], dict[str, object]]:
    first_date = enhanced.index.min()
    relative_columns = [relative_momentum_6m_col(ticker) for ticker in sectors]
    first_relative_values = enhanced.loc[first_date, relative_columns]
    relative_sum = float(first_relative_values.sum())
    relative_mean = float(first_relative_values.mean())
    if abs(relative_sum) > 1e-12 or abs(relative_mean) > 1e-12:
        raise ValueError(
            f"{config_name} relative momentum does not sum to zero on first valid date."
        )
    return (
        [f"{config_name}: first-date relative momentum sum and mean are approximately zero"],
        {
            "first_valid_enhanced_feature_date": first_date.date().isoformat(),
            "relative_momentum_sum": relative_sum,
            "relative_momentum_mean": relative_mean,
        },
    )


def manual_spot_check(
    enhanced: pd.DataFrame,
    sector_returns: pd.DataFrame,
    sectors: list[str],
    config_name: str,
) -> dict[str, object]:
    date = enhanced.index.min()
    window_3 = sector_returns.loc[:date, "XLB"].tail(3)
    window_6 = sector_returns.loc[:date, "XLB"].tail(6)

    manual_3m = compounded_return(window_3)
    manual_6m = compounded_return(window_6)
    manual_vol = float(window_6.std() * (12**0.5))

    universe_momentum_6m = {
        ticker: compounded_return(sector_returns.loc[:date, ticker].tail(6))
        for ticker in sectors
    }
    manual_relative = manual_6m - sum(universe_momentum_6m.values()) / len(sectors)

    checks = {
        "manual_xlb_momentum_3m": manual_3m,
        "saved_xlb_momentum_3m": enhanced.at[date, momentum_3m_col("XLB")],
        "manual_xlb_momentum_6m": manual_6m,
        "saved_xlb_momentum_6m": enhanced.at[date, momentum_6m_col("XLB")],
        "manual_xlb_volatility_6m": manual_vol,
        "saved_xlb_volatility_6m": enhanced.at[date, volatility_6m_col("XLB")],
        "manual_xlb_relative_momentum_6m": manual_relative,
        "saved_xlb_relative_momentum_6m": enhanced.at[
            date, relative_momentum_6m_col("XLB")
        ],
    }

    for manual_key, saved_key in [
        ("manual_xlb_momentum_3m", "saved_xlb_momentum_3m"),
        ("manual_xlb_momentum_6m", "saved_xlb_momentum_6m"),
        ("manual_xlb_volatility_6m", "saved_xlb_volatility_6m"),
        ("manual_xlb_relative_momentum_6m", "saved_xlb_relative_momentum_6m"),
    ]:
        if abs(checks[manual_key] - checks[saved_key]) > 1e-12:
            raise ValueError(f"{config_name} manual spot check failed for {manual_key}.")

    return {
        "scope": config_name,
        "first_valid_enhanced_feature_date": date.date().isoformat(),
        **checks,
    }


def validate_output(
    enhanced: pd.DataFrame,
    original: pd.DataFrame,
    sector_returns: pd.DataFrame,
    config: ScopeConfig,
) -> list[str]:
    expected_enhanced = enhanced_feature_columns(config.sectors)
    missing_enhanced = sorted(set(expected_enhanced) - set(enhanced.columns))
    if missing_enhanced:
        raise ValueError(
            f"{config.name} missing enhanced columns: {', '.join(missing_enhanced)}"
        )
    if enhanced.empty:
        raise ValueError(f"{config.name} enhanced dataset is empty.")
    if enhanced.index.has_duplicates:
        raise ValueError(f"{config.name} enhanced dataset has duplicate dates.")
    if not enhanced.index.is_monotonic_increasing:
        raise ValueError(f"{config.name} enhanced dataset dates are not sorted.")
    if enhanced.isna().any().any():
        raise ValueError(f"{config.name} enhanced dataset contains missing values.")

    validate_no_current_incomplete_month(enhanced, f"{config.name} enhanced dataset")
    validate_targets_unchanged(original, enhanced, config.sectors, config.name)
    validate_next_month_targets(enhanced, sector_returns, config.sectors, config.name)

    validations = validate_no_future_feature_use(
        enhanced,
        sector_returns,
        config.sectors,
        config.name,
    )
    relative_validations, relative_zero_sum = validate_relative_momentum_zero_sum(
        enhanced,
        config.sectors,
        config.name,
    )
    validations.extend(relative_validations)
    validations.extend(
        [
            f"{config.name}: output dataset is non-empty",
            f"{config.name}: dates are sorted and unique",
            f"{config.name}: no missing values in enhanced output",
            f"{config.name}: target columns exactly match Step 4 dataset",
            f"{config.name}: current incomplete calendar month is absent",
            f"{config.name}: expected enhanced feature columns exist",
        ]
    )
    return validations, relative_zero_sum


def column_structure_audit(
    enhanced: pd.DataFrame,
    config: ScopeConfig,
) -> dict[str, object]:
    enhanced_columns = enhanced_feature_columns(config.sectors)
    target_return_columns = [target_return_col(ticker) for ticker in config.sectors]
    target_rank_columns = [target_rank_col(ticker) for ticker in config.sectors]
    modeled_field_count = len(enhanced.columns)
    saved_csv_column_count = modeled_field_count + 1
    return {
        "complete_saved_csv_column_count": saved_csv_column_count,
        "modeled_field_count_excluding_date_metadata": modeled_field_count,
        "existing_step4_macro_feature_count": len(MACRO_FEATURES),
        "existing_target_return_count": len(target_return_columns),
        "existing_target_rank_count": len(target_rank_columns),
        "target_best_sector_count": 1,
        "enhanced_momentum_volatility_feature_count": len(enhanced_columns),
        "metadata_or_date_column_count": 1,
        "metadata_or_helper_columns": "date",
        "one_column_difference_explained_by": "date",
        "extra_column_classification": "required date identifier metadata; never a model input feature",
    }


def print_audit(
    sector_returns: pd.DataFrame,
    scope_results: list[dict[str, object]],
) -> None:
    print("Enhanced Model Dataset Build Audit")
    print("==================================")
    print()
    print("Enhanced feature definitions:")
    definitions = pd.DataFrame(
        [
            {
                "feature_pattern": "sector_momentum_3m_<ticker>",
                "definition": "Compounded realized sector return over months t-2 through t",
            },
            {
                "feature_pattern": "sector_momentum_6m_<ticker>",
                "definition": "Compounded realized sector return over months t-5 through t",
            },
            {
                "feature_pattern": "sector_relative_momentum_6m_<ticker>",
                "definition": "Sector 6-month momentum minus same-date universe average",
            },
            {
                "feature_pattern": "sector_volatility_6m_<ticker>",
                "definition": "6-month trailing monthly return standard deviation annualized by sqrt(12)",
            },
        ]
    )
    print(definitions.to_string(index=False))
    print()
    print(
        "Sector returns input: "
        f"{sector_returns.index.min().date()} to {sector_returns.index.max().date()}, "
        f"shape {sector_returns.shape}"
    )
    print()

    for result in scope_results:
        config = result["config"]
        model_data = result["model_data"]
        enhanced = result["enhanced"]
        print(f"Scope: {config.name}")
        print("-" * (7 + len(config.name)))
        print(f"Sector universe for relative momentum: {', '.join(config.sectors)}")
        print(
            f"Model input: {model_data.index.min().date()} to "
            f"{model_data.index.max().date()}, shape {model_data.shape}"
        )
        print(
            f"Enhanced output: {enhanced.index.min().date()} to "
            f"{enhanced.index.max().date()}, shape {enhanced.shape}"
        )
        print(f"First valid enhanced feature date: {enhanced.index.min().date()}")
        print(f"Last valid enhanced feature date: {enhanced.index.max().date()}")
        print("Rows removed:")
        print(pd.DataFrame([result["removal_summary"]]).to_string(index=False))
        print("Column structure audit:")
        print(pd.DataFrame([result["column_audit"]]).to_string(index=False))
        print("XLB enhanced feature columns:")
        print(
            pd.DataFrame(
                [
                    {
                        "sector": "XLB",
                        "enhanced_feature_column": column,
                    }
                    for column in [
                        momentum_3m_col("XLB"),
                        momentum_6m_col("XLB"),
                        relative_momentum_6m_col("XLB"),
                        volatility_6m_col("XLB"),
                    ]
                ]
            ).to_string(index=False)
        )
        print("Manual XLB spot-check:")
        spot = pd.DataFrame([result["spot_check"]]).drop(columns=["scope"])
        print(spot.to_string(index=False))
        print("Cross-sectional relative momentum zero-sum check:")
        print(pd.DataFrame([result["relative_zero_sum"]]).to_string(index=False))
        print("Leakage and output validation:")
        for validation in result["validations"]:
            print(f"- {validation}")
        print(f"Missing-value count: {int(enhanced.isna().sum().sum())}")
        print(f"Output file: {config.output_path.relative_to(PROJECT_ROOT)}")
        print()


def main() -> None:
    sector_returns = read_sector_returns()
    scope_results = []

    for config in SCOPES:
        model_data = read_model_dataset(config)
        trailing_features = build_trailing_features(sector_returns, config.sectors)
        enhanced, removal_summary = merge_enhanced_dataset(
            model_data,
            trailing_features,
            config,
        )
        validations, relative_zero_sum = validate_output(
            enhanced,
            model_data,
            sector_returns,
            config,
        )
        spot_check = manual_spot_check(
            enhanced,
            sector_returns,
            config.sectors,
            config.name,
        )
        column_audit = column_structure_audit(enhanced, config)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.to_csv(config.output_path, index_label="date")
        scope_results.append(
            {
                "config": config,
                "model_data": model_data,
                "enhanced": enhanced,
                "removal_summary": removal_summary,
                "validations": validations,
                "spot_check": spot_check,
                "relative_zero_sum": relative_zero_sum,
                "column_audit": column_audit,
            }
        )

    print_audit(sector_returns, scope_results)


if __name__ == "__main__":
    main()
