"""Download and audit FRED macroeconomic data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


START_DATE = "1998-01-01"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "INDPRO": {
        "name": "Industrial Production Index",
        "rule": "reported monthly observation labeled at calendar month-end",
        "frequency": "monthly",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "rule": "reported monthly observation labeled at calendar month-end",
        "frequency": "monthly",
    },
    "CPIAUCSL": {
        "name": "Consumer Price Index",
        "rule": "reported monthly observation labeled at calendar month-end",
        "frequency": "monthly",
    },
    "FEDFUNDS": {
        "name": "Federal Funds Effective Rate",
        "rule": "reported monthly observation labeled at calendar month-end",
        "frequency": "monthly",
    },
    "GS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "rule": "reported monthly observation labeled at calendar month-end",
        "frequency": "monthly",
    },
    "T10Y2Y": {
        "name": "10-Year Treasury minus 2-Year Treasury Spread",
        "rule": "last available non-missing observation in completed calendar month",
        "frequency": "daily",
    },
    "BAA10Y": {
        "name": "Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity",
        "rule": "last available non-missing observation in completed calendar month",
        "frequency": "daily",
    },
    "M2SL": {
        "name": "M2 Money Stock",
        "rule": "reported monthly observation labeled at calendar month-end",
        "frequency": "monthly",
    },
    "VIXCLS": {
        "name": "CBOE Volatility Index",
        "rule": "last available non-missing observation in completed calendar month",
        "frequency": "daily",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT = PROJECT_ROOT / "data" / "raw" / "macro_fred_raw.csv"
MONTHLY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "macro_monthly.csv"


def current_month_is_incomplete(as_of_date: pd.Timestamp) -> bool:
    """Return True until the current calendar month has ended."""
    month_end = as_of_date + pd.offsets.MonthEnd(0)
    return as_of_date.normalize() < month_end.normalize()


def load_api_key() -> str:
    """Load the FRED API key without printing it."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is missing. Add it to the local .env file before running."
        )
    return api_key


def fetch_series(series_id: str, api_key: str) -> pd.DataFrame:
    """Fetch one FRED series from the official observations endpoint."""
    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE,
    }

    try:
        response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"FRED HTTP request failed for {series_id}: {exc}") from exc

    payload = response.json()
    if "error_code" in payload:
        message = payload.get("error_message", "unknown FRED API error")
        raise RuntimeError(f"FRED API error for {series_id}: {message}")

    observations = payload.get("observations")
    if not observations:
        raise RuntimeError(f"FRED API returned no observations for {series_id}.")

    data = pd.DataFrame(observations)
    expected_columns = {"date", "value"}
    if not expected_columns.issubset(data.columns):
        raise RuntimeError(f"FRED response for {series_id} is missing required fields.")

    data = data.loc[:, ["date", "value"]].copy()
    data["series_id"] = series_id
    data["date"] = pd.to_datetime(data["date"])
    data["value"] = data["value"].replace(".", pd.NA)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    return data.loc[:, ["date", "series_id", "value"]]


def download_raw_data(api_key: str) -> pd.DataFrame:
    """Download all expected FRED series in long format."""
    frames = [fetch_series(series_id, api_key) for series_id in SERIES]
    raw_data = pd.concat(frames, ignore_index=True)
    raw_data = raw_data.sort_values(["series_id", "date"]).reset_index(drop=True)

    if raw_data.empty:
        raise ValueError("Raw FRED dataset is empty.")

    downloaded_series = set(raw_data["series_id"].unique())
    missing_series = sorted(set(SERIES) - downloaded_series)
    if missing_series:
        raise ValueError(f"Missing expected FRED series: {', '.join(missing_series)}")

    return raw_data


def completed_month_cutoff(as_of_date: pd.Timestamp) -> pd.Timestamp:
    """Return the latest calendar month-end that is complete as of the run date."""
    if current_month_is_incomplete(as_of_date):
        return as_of_date.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
    return as_of_date.to_period("M").to_timestamp("M")


def process_monthly_series(series_data: pd.DataFrame, frequency: str) -> pd.Series:
    """Convert one raw FRED series to completed-month frequency."""
    data = series_data.sort_values("date").copy()

    if frequency == "monthly":
        data["month_end"] = data["date"].dt.to_period("M").dt.to_timestamp("M")
        monthly = data.drop_duplicates("month_end", keep="last").set_index("month_end")
        return monthly["value"]

    if frequency == "daily":
        data["month_end"] = data["date"].dt.to_period("M").dt.to_timestamp("M")
        return data.groupby("month_end")["value"].agg(
            lambda values: values.dropna().iloc[-1] if values.notna().any() else pd.NA
        )

    raise ValueError(f"Unsupported source frequency: {frequency}")


def build_monthly_data(
    raw_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    """Build a wide completed-month macro dataset."""
    monthly_parts = {}
    cutoff = completed_month_cutoff(as_of_date)
    excluded_incomplete_month = False

    for series_id, metadata in SERIES.items():
        series_data = raw_data.loc[raw_data["series_id"] == series_id]
        monthly_parts[series_id] = process_monthly_series(
            series_data,
            metadata["frequency"],
        )

    monthly = pd.DataFrame(monthly_parts).sort_index()
    current_period = as_of_date.to_period("M")
    if current_month_is_incomplete(as_of_date):
        incomplete_mask = monthly.index.to_period("M") == current_period
        excluded_incomplete_month = bool(incomplete_mask.any())
        monthly = monthly.loc[~incomplete_mask]

    monthly = monthly.loc[monthly.index <= cutoff]
    monthly.index.name = "date"

    return monthly, excluded_incomplete_month


def validate_data(
    raw_data: pd.DataFrame,
    monthly_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> None:
    """Validate raw and processed macro datasets."""
    if raw_data.empty:
        raise ValueError("Raw FRED dataset is empty.")

    missing_raw_series = sorted(set(SERIES) - set(raw_data["series_id"].unique()))
    if missing_raw_series:
        raise ValueError(
            "Missing expected downloaded series: " + ", ".join(missing_raw_series)
        )

    if not raw_data.sort_values(["series_id", "date"]).index.equals(raw_data.index):
        raise ValueError("Raw FRED data is not sorted by series_id and date.")

    if monthly_data.empty:
        raise ValueError("Processed monthly macro dataset is empty.")

    if not monthly_data.index.is_monotonic_increasing:
        raise ValueError("Processed monthly dates are not sorted.")

    if monthly_data.index.has_duplicates:
        raise ValueError("Processed monthly dataset contains duplicate dates.")

    if current_month_is_incomplete(as_of_date):
        current_period = as_of_date.to_period("M")
        if (monthly_data.index.to_period("M") == current_period).any():
            raise ValueError(
                "Processed monthly dataset includes the current incomplete month."
            )

    missing_columns = sorted(set(SERIES) - set(monthly_data.columns))
    if missing_columns:
        raise ValueError(
            "Processed monthly dataset is missing columns: " + ", ".join(missing_columns)
        )


def build_audit_table(
    raw_data: pd.DataFrame,
    monthly_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-series audit metrics."""
    rows = []

    for series_id, metadata in SERIES.items():
        raw_series = raw_data.loc[raw_data["series_id"] == series_id].dropna(
            subset=["value"]
        )
        processed_series = monthly_data[series_id].dropna()

        rows.append(
            {
                "series_id": series_id,
                "indicator_name": metadata["name"],
                "source_frequency": metadata["frequency"],
                "frequency_handling_rule": metadata["rule"],
                "first_valid_raw_date": (
                    raw_series["date"].min().date().isoformat()
                    if not raw_series.empty
                    else None
                ),
                "latest_raw_row_date": (
                    raw_data.loc[raw_data["series_id"] == series_id, "date"]
                    .max()
                    .date()
                    .isoformat()
                ),
                "latest_valid_raw_value_date": (
                    raw_series["date"].max().date().isoformat()
                    if not raw_series.empty
                    else None
                ),
                "valid_raw_observation_count": int(raw_series["value"].count()),
                "first_valid_processed_month": (
                    processed_series.index.min().date().isoformat()
                    if not processed_series.empty
                    else None
                ),
                "last_valid_processed_month": (
                    processed_series.index.max().date().isoformat()
                    if not processed_series.empty
                    else None
                ),
                "processed_non_missing_count": int(processed_series.count()),
                "processed_missing_count": int(monthly_data[series_id].isna().sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    api_key = load_api_key()
    as_of_date = pd.Timestamp.today().normalize()

    raw_data = download_raw_data(api_key)
    monthly_data, excluded_incomplete_month = build_monthly_data(raw_data, as_of_date)
    validate_data(raw_data, monthly_data, as_of_date)

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MONTHLY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    raw_data.to_csv(RAW_OUTPUT, index=False)
    monthly_data.to_csv(MONTHLY_OUTPUT, index_label="date")

    audit_table = build_audit_table(raw_data, monthly_data)
    latest_raw_observation_date = raw_data["date"].max()
    latest_completed_processed_month = monthly_data.index.max()
    total_missing_values = int(monthly_data.isna().sum().sum())

    print("FRED Macro Data Download Audit")
    print("=" * 30)
    print(f"Start date requested: {START_DATE}")
    print(f"Latest raw observation date: {latest_raw_observation_date.date().isoformat()}")
    print(
        "Latest completed processed month: "
        f"{latest_completed_processed_month.date().isoformat()}"
    )
    print(
        "Incomplete current-month row excluded: "
        f"{'yes' if excluded_incomplete_month else 'no'}"
    )
    print(f"Processed monthly dataset shape: {monthly_data.shape}")
    print(f"Total missing-value count: {total_missing_values}")
    print()
    print(audit_table.to_string(index=False))
    print()
    print("Saved datasets:")
    print(f"- {RAW_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"- {MONTHLY_OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
