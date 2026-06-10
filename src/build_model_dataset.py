"""Build model-ready monthly macro feature and sector return datasets."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACRO_INPUT = PROJECT_ROOT / "data" / "processed" / "macro_monthly.csv"
SECTOR_RETURNS_INPUT = PROJECT_ROOT / "data" / "processed" / "sector_returns_monthly.csv"
OUTPUT_11_SECTOR = PROJECT_ROOT / "data" / "processed" / "model_dataset_11_sector.csv"
OUTPUT_9_SECTOR = (
    PROJECT_ROOT / "data" / "processed" / "model_dataset_9_sector_long_history.csv"
)

MACRO_COLUMNS = [
    "INDPRO",
    "UNRATE",
    "CPIAUCSL",
    "FEDFUNDS",
    "GS10",
    "T10Y2Y",
    "BAA10Y",
    "M2SL",
    "VIXCLS",
]

ALL_SECTOR_ETFS = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

LONG_HISTORY_SECTOR_ETFS = [
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]

FEATURE_SPECS = {
    "indpro_yoy": {
        "source": "INDPRO",
        "lag_rule": "shifted 1 calendar month",
        "description": "INDPRO 12-month percentage change * 100",
    },
    "unemployment_change_3m": {
        "source": "UNRATE",
        "lag_rule": "shifted 1 calendar month",
        "description": "UNRATE 3-month difference",
    },
    "inflation_yoy": {
        "source": "CPIAUCSL",
        "lag_rule": "shifted 1 calendar month",
        "description": "CPIAUCSL 12-month percentage change * 100",
    },
    "m2_yoy": {
        "source": "M2SL",
        "lag_rule": "shifted 1 calendar month",
        "description": "M2SL 12-month percentage change * 100",
    },
    "fedfunds_level": {
        "source": "FEDFUNDS",
        "lag_rule": "shifted 1 calendar month",
        "description": "FEDFUNDS level",
    },
    "treasury_10y_level": {
        "source": "GS10",
        "lag_rule": "shifted 1 calendar month",
        "description": "GS10 level",
    },
    "yield_curve_spread": {
        "source": "T10Y2Y",
        "lag_rule": "current completed month-end observation",
        "description": "T10Y2Y level",
    },
    "credit_spread": {
        "source": "BAA10Y",
        "lag_rule": "current completed month-end observation",
        "description": "BAA10Y level",
    },
    "vix_level": {
        "source": "VIXCLS",
        "lag_rule": "current completed month-end observation",
        "description": "VIXCLS level",
    },
}


def current_month_is_incomplete(as_of_date: pd.Timestamp) -> bool:
    """Return True until the current calendar month has ended."""
    return as_of_date.normalize() < (
        as_of_date + pd.offsets.MonthEnd(0)
    ).normalize()


def read_monthly_csv(path: Path, date_column: str, required_columns: list[str]) -> pd.DataFrame:
    """Read and validate a monthly CSV indexed by date."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")

    data = pd.read_csv(path, parse_dates=[date_column])
    missing_columns = sorted(set([date_column, *required_columns]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing_columns)}")

    data = data.loc[:, [date_column, *required_columns]].copy()
    data = data.rename(columns={date_column: "date"}).set_index("date")
    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError(f"{path.name} contains duplicate dates.")

    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} dates are not sorted.")

    return data


def validate_no_incomplete_current_month(data: pd.DataFrame, name: str) -> None:
    """Reject datasets containing the current incomplete calendar month."""
    as_of_date = pd.Timestamp.today().normalize()
    if current_month_is_incomplete(as_of_date):
        current_period = as_of_date.to_period("M")
        if (data.index.to_period("M") == current_period).any():
            raise ValueError(f"{name} includes the current incomplete calendar month.")


def build_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Build macro features and apply conservative availability lags.

    Release lags reduce look-ahead bias: many monthly economic series are not known at
    the month-end label date, so this dataset uses the prior completed month's value.
    Standard FRED downloads may include revised historical values; a future advanced
    version could use ALFRED vintage data for stricter point-in-time analysis.
    """
    raw_features = pd.DataFrame(index=macro.index)
    raw_features["indpro_yoy"] = macro["INDPRO"].pct_change(
        periods=12, fill_method=None
    ) * 100
    raw_features["unemployment_change_3m"] = macro["UNRATE"].diff(periods=3)
    raw_features["inflation_yoy"] = macro["CPIAUCSL"].pct_change(
        periods=12, fill_method=None
    ) * 100
    raw_features["m2_yoy"] = macro["M2SL"].pct_change(periods=12, fill_method=None) * 100
    raw_features["fedfunds_level"] = macro["FEDFUNDS"]
    raw_features["treasury_10y_level"] = macro["GS10"]
    raw_features["yield_curve_spread"] = macro["T10Y2Y"]
    raw_features["credit_spread"] = macro["BAA10Y"]
    raw_features["vix_level"] = macro["VIXCLS"]

    features = raw_features.copy()
    lagged_features = [
        "indpro_yoy",
        "unemployment_change_3m",
        "inflation_yoy",
        "m2_yoy",
        "fedfunds_level",
        "treasury_10y_level",
    ]
    features.loc[:, lagged_features] = features.loc[:, lagged_features].shift(1)

    return features


def forward_fill_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-fill feature gaps for at most one month and audit each filled cell."""
    filled = features.ffill(limit=1)
    filled_mask = features.isna() & filled.notna()

    rows = []
    for date, feature in zip(*filled_mask.to_numpy().nonzero(), strict=False):
        index_date = filled_mask.index[date]
        feature_name = filled_mask.columns[feature]
        previous_valid_date = features.loc[:index_date, feature_name].iloc[:-1].last_valid_index()
        rows.append(
            {
                "date": index_date.date().isoformat(),
                "feature": feature_name,
                "filled_value": filled.at[index_date, feature_name],
                "source_date": previous_valid_date.date().isoformat()
                if previous_valid_date is not None
                else None,
                "reason": "one-month limited forward-fill of most recently available published value",
            }
        )

    return filled, pd.DataFrame(rows)


def add_targets(
    base: pd.DataFrame,
    sector_returns: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Align features at month t with sector returns at month t+1."""
    target_returns = sector_returns.loc[:, tickers].shift(-1)
    target_returns = target_returns.rename(
        columns={ticker: f"target_next_return_{ticker}" for ticker in tickers}
    )

    target_ranks = target_returns.rank(axis=1, ascending=False, method="min")
    target_ranks = target_ranks.rename(
        columns={
            f"target_next_return_{ticker}": f"target_next_rank_{ticker}"
            for ticker in tickers
        }
    )

    result = base.join(target_returns, how="inner").join(target_ranks, how="inner")
    target_best_sector = pd.Series(pd.NA, index=target_returns.index, dtype="object")
    rows_with_targets = target_returns.notna().any(axis=1)
    target_best_sector.loc[rows_with_targets] = target_returns.loc[
        rows_with_targets
    ].idxmax(axis=1)
    result["target_best_sector"] = target_best_sector.str.replace(
        "target_next_return_",
        "",
        regex=False,
    )

    return result


def filter_model_dataset(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    tickers: list[str],
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Require complete features and complete target returns for a ticker universe."""
    target_return_columns = [f"target_next_return_{ticker}" for ticker in tickers]
    target_rank_columns = [f"target_next_rank_{ticker}" for ticker in tickers]

    start_rows = len(dataset)
    missing_features = dataset[feature_columns].isna().any(axis=1)
    missing_targets = dataset[target_return_columns].isna().any(axis=1)
    both_missing = missing_features & missing_targets
    removed = missing_features | missing_targets
    filtered = dataset.loc[~removed].copy()

    removal_summary = {
        "starting_rows": start_rows,
        "rows_with_missing_macro_features": int(missing_features.sum()),
        "rows_with_missing_next_month_targets": int(missing_targets.sum()),
        "rows_with_both_missing_features_and_targets": int(both_missing.sum()),
        "total_unique_rows_removed": int(removed.sum()),
        "final_rows_retained": len(filtered),
    }
    if start_rows - removal_summary["total_unique_rows_removed"] != len(filtered):
        raise ValueError("Row-removal accounting does not reconcile.")

    diagnostic_rows = []
    for date, row in dataset.loc[missing_features].iterrows():
        missing_feature_names = [
            feature for feature in feature_columns if pd.isna(row[feature])
        ]
        row_missing_targets = bool(missing_targets.loc[date])
        if row_missing_targets:
            reason = "excluded: missing macro features and missing next-month targets"
        else:
            reason = "excluded: missing macro features"
        diagnostic_rows.append(
            {
                "feature_date": date.date().isoformat(),
                "missing_feature_names": ", ".join(missing_feature_names),
                "also_missing_next_month_targets": row_missing_targets,
                "exclusion_reason": reason,
            }
        )

    selected_columns = [
        *feature_columns,
        *target_return_columns,
        *target_rank_columns,
        "target_best_sector",
    ]
    return filtered.loc[:, selected_columns], removal_summary, pd.DataFrame(diagnostic_rows)


def validate_targets(
    dataset: pd.DataFrame,
    sector_returns: pd.DataFrame,
    tickers: list[str],
    dataset_name: str,
) -> list[str]:
    """Validate target alignment, best sector, ranks, and dates."""
    if dataset.empty:
        raise ValueError(f"{dataset_name} is empty.")

    if not dataset.index.is_monotonic_increasing:
        raise ValueError(f"{dataset_name} dates are not sorted.")

    if dataset.index.has_duplicates:
        raise ValueError(f"{dataset_name} contains duplicate dates.")

    for date, row in dataset.iterrows():
        next_month = date + pd.offsets.MonthEnd(1)
        if next_month not in sector_returns.index:
            raise ValueError(f"{dataset_name} target month missing for {date.date()}.")

        target_returns = {
            ticker: row[f"target_next_return_{ticker}"] for ticker in tickers
        }
        original_returns = sector_returns.loc[next_month, tickers]
        for ticker, target_value in target_returns.items():
            original_value = original_returns[ticker]
            if pd.isna(target_value) or pd.isna(original_value) or abs(target_value - original_value) > 1e-12:
                raise ValueError(
                    f"{dataset_name} target mismatch for {ticker} at {date.date()}."
                )

        best_sector = max(target_returns, key=target_returns.get)
        if row["target_best_sector"] != best_sector:
            raise ValueError(f"{dataset_name} target_best_sector mismatch at {date.date()}.")

        if row[f"target_next_rank_{best_sector}"] != 1:
            raise ValueError(f"{dataset_name} rank 1 mismatch at {date.date()}.")

    return [
        f"{dataset_name}: non-empty",
        f"{dataset_name}: dates sorted and unique",
        f"{dataset_name}: targets equal original next-month ETF returns",
        f"{dataset_name}: target_best_sector agrees with highest target return",
        f"{dataset_name}: rank 1 agrees with target_best_sector",
    ]


def print_audit(
    macro: pd.DataFrame,
    sector_returns: pd.DataFrame,
    features_before_fill: pd.DataFrame,
    features: pd.DataFrame,
    forward_fill_audit: pd.DataFrame,
    output_11: pd.DataFrame,
    output_9: pd.DataFrame,
    removals_11: dict[str, int],
    removals_9: dict[str, int],
    missing_feature_rows_11: pd.DataFrame,
    missing_feature_rows_9: pd.DataFrame,
    validation_results: list[str],
) -> None:
    """Print a detailed build audit."""
    feature_audit = pd.DataFrame(
        [
            {
                "feature": feature,
                "source": metadata["source"],
                "lag_rule": metadata["lag_rule"],
                "description": metadata["description"],
            }
            for feature, metadata in FEATURE_SPECS.items()
        ]
    )

    print("Model Dataset Build Audit")
    print("=========================")
    print()
    print("Feature rules:")
    print(feature_audit.to_string(index=False))
    print()
    print("Input date ranges:")
    print(f"- macro_monthly: {macro.index.min().date()} to {macro.index.max().date()}, shape {macro.shape}")
    print(
        "- sector_returns_monthly: "
        f"{sector_returns.index.min().date()} to {sector_returns.index.max().date()}, "
        f"shape {sector_returns.shape}"
    )
    print()
    print("Rows with missing macro features before dataset filtering:")
    if missing_feature_rows_11.empty:
        print("- 11-sector: none")
    else:
        table = missing_feature_rows_11.copy()
        table.insert(0, "dataset_scope", "11-sector")
        print(table.to_string(index=False))
    if missing_feature_rows_9.empty:
        print("- 9-sector: none")
    else:
        table = missing_feature_rows_9.copy()
        table.insert(0, "dataset_scope", "9-sector")
        print(table.to_string(index=False))
    print()
    reconciliation = pd.DataFrame(
        [
            {"dataset_scope": "11-sector", **removals_11},
            {"dataset_scope": "9-sector", **removals_9},
        ]
    )
    reconciliation["accounting_reconciles"] = (
        reconciliation["starting_rows"]
        - reconciliation["total_unique_rows_removed"]
        == reconciliation["final_rows_retained"]
    )
    print("Row-removal reconciliation:")
    print(reconciliation.to_string(index=False))
    print()
    output_audit = pd.DataFrame(
        [
            {
                "dataset_scope": "11-sector",
                "output_start": output_11.index.min().date().isoformat(),
                "output_end": output_11.index.max().date().isoformat(),
                "rows": output_11.shape[0],
                "columns": output_11.shape[1],
                "missing_value_count": int(output_11.isna().sum().sum()),
                "duplicate_date_count": int(output_11.index.duplicated().sum()),
                "first_usable_prediction_month": (
                    output_11.index.min() + pd.offsets.MonthEnd(1)
                ).date().isoformat(),
                "last_usable_prediction_month": (
                    output_11.index.max() + pd.offsets.MonthEnd(1)
                ).date().isoformat(),
            },
            {
                "dataset_scope": "9-sector",
                "output_start": output_9.index.min().date().isoformat(),
                "output_end": output_9.index.max().date().isoformat(),
                "rows": output_9.shape[0],
                "columns": output_9.shape[1],
                "missing_value_count": int(output_9.isna().sum().sum()),
                "duplicate_date_count": int(output_9.index.duplicated().sum()),
                "first_usable_prediction_month": (
                    output_9.index.min() + pd.offsets.MonthEnd(1)
                ).date().isoformat(),
                "last_usable_prediction_month": (
                    output_9.index.max() + pd.offsets.MonthEnd(1)
                ).date().isoformat(),
            },
        ]
    )
    print("Final output audit:")
    print(output_audit.to_string(index=False))
    print()
    print("Forward-filled feature cells:")
    if forward_fill_audit.empty:
        print("none")
    else:
        print(forward_fill_audit.to_string(index=False))
        print(
            "Forward-fill validation: all source dates are earlier than the filled "
            "feature dates, so no future values are used."
        )
    print()
    missing_before = features_before_fill.isna().sum()
    missing_after = features.isna().sum()
    missing_summary = pd.DataFrame(
        {
            "missing_before_forward_fill": missing_before,
            "missing_after_forward_fill": missing_after,
        }
    )
    print("Remaining feature missing-value counts:")
    print(missing_summary.to_string())
    print()
    print("Validation results:")
    for result in validation_results:
        print(f"- {result}")
    print()
    print("Output files:")
    print(f"- {OUTPUT_11_SECTOR.relative_to(PROJECT_ROOT)}")
    print(f"- {OUTPUT_9_SECTOR.relative_to(PROJECT_ROOT)}")


def main() -> None:
    macro = read_monthly_csv(MACRO_INPUT, "date", MACRO_COLUMNS)
    sector_returns = read_monthly_csv(SECTOR_RETURNS_INPUT, "Date", ALL_SECTOR_ETFS)

    validate_no_incomplete_current_month(macro, "macro_monthly.csv")
    validate_no_incomplete_current_month(sector_returns, "sector_returns_monthly.csv")

    features_before_fill = build_macro_features(macro)
    features, forward_fill_audit = forward_fill_features(features_before_fill)
    if features.isna().sum().sum() > 0:
        warnings.warn(
            "Missing values remain in the full macro feature history after one-month "
            "limited forward-fill. Any aligned rows with incomplete features will be "
            "excluded from model-ready outputs.",
            stacklevel=2,
        )

    feature_columns = list(FEATURE_SPECS)
    base = features.loc[:, feature_columns]

    dataset_11 = add_targets(base, sector_returns, ALL_SECTOR_ETFS)
    output_11, removals_11, missing_feature_rows_11 = filter_model_dataset(
        dataset_11,
        feature_columns,
        ALL_SECTOR_ETFS,
    )

    dataset_9 = add_targets(base, sector_returns, LONG_HISTORY_SECTOR_ETFS)
    output_9, removals_9, missing_feature_rows_9 = filter_model_dataset(
        dataset_9,
        feature_columns,
        LONG_HISTORY_SECTOR_ETFS,
    )

    validation_results = []
    validation_results.extend(
        validate_targets(output_11, sector_returns, ALL_SECTOR_ETFS, "11-sector dataset")
    )
    validation_results.extend(
        validate_targets(
            output_9,
            sector_returns,
            LONG_HISTORY_SECTOR_ETFS,
            "9-sector dataset",
        )
    )

    OUTPUT_11_SECTOR.parent.mkdir(parents=True, exist_ok=True)
    output_11.to_csv(OUTPUT_11_SECTOR, index_label="date")
    output_9.to_csv(OUTPUT_9_SECTOR, index_label="date")

    print_audit(
        macro,
        sector_returns,
        features_before_fill,
        features,
        forward_fill_audit,
        output_11,
        output_9,
        removals_11,
        removals_9,
        missing_feature_rows_11,
        missing_feature_rows_9,
        validation_results,
    )


if __name__ == "__main__":
    main()
