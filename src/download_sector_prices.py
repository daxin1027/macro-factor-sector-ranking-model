"""Download and audit adjusted prices for U.S. sector ETFs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


START_DATE = "2000-01-01"

SECTOR_ETFS = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT = PROJECT_ROOT / "data" / "raw" / "sector_prices_daily.csv"
MONTHLY_PRICES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sector_prices_monthly.csv"
MONTHLY_RETURNS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sector_returns_monthly.csv"


def current_month_is_incomplete(as_of_date: pd.Timestamp) -> bool:
    """Return True until the current calendar month has ended."""
    month_end = as_of_date + pd.offsets.MonthEnd(0)
    return as_of_date.normalize() < month_end.normalize()


def keep_completed_months_only(
    monthly_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    """Drop the current month when it is still an incomplete calendar month."""
    if not current_month_is_incomplete(as_of_date):
        return monthly_prices, False

    current_period = as_of_date.to_period("M")
    incomplete_month_mask = monthly_prices.index.to_period("M") == current_period
    excluded_incomplete_month = bool(incomplete_month_mask.any())

    return monthly_prices.loc[~incomplete_month_mask], excluded_incomplete_month


def extract_adjusted_close(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close prices from yfinance output."""
    if downloaded.empty:
        raise RuntimeError("Download failed: yfinance returned an empty DataFrame.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        level_0 = downloaded.columns.get_level_values(0)
        level_1 = downloaded.columns.get_level_values(1)

        if "Adj Close" in level_0:
            prices = downloaded["Adj Close"]
        elif "Adj Close" in level_1:
            prices = downloaded.xs("Adj Close", axis=1, level=1)
        else:
            raise RuntimeError(
                "Download failed: yfinance output does not contain an 'Adj Close' field."
            )
    elif "Adj Close" in downloaded.columns:
        prices = downloaded[["Adj Close"]]
        if len(tickers) == 1:
            prices.columns = tickers
    else:
        raise RuntimeError(
            "Download failed: yfinance output does not contain an 'Adj Close' field."
        )

    prices = prices.copy()
    prices.columns = [str(column).upper() for column in prices.columns]
    missing_columns = sorted(set(tickers) - set(prices.columns))
    if missing_columns:
        raise RuntimeError(
            "Download failed: missing adjusted close columns for "
            + ", ".join(missing_columns)
            + "."
        )

    return prices.loc[:, tickers]


def validate_prices(
    daily_prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    tickers: list[str],
    as_of_date: pd.Timestamp,
) -> None:
    """Run basic checks on downloaded and transformed data."""
    missing_columns = sorted(set(tickers) - set(daily_prices.columns))
    if missing_columns:
        raise ValueError(f"Missing expected ETF columns: {', '.join(missing_columns)}")

    if not daily_prices.index.is_monotonic_increasing:
        raise ValueError("Daily price dates are not sorted chronologically.")

    if daily_prices.index.has_duplicates:
        raise ValueError("Daily price data contains duplicate dates.")

    if monthly_prices.empty:
        raise ValueError("Monthly price dataset is empty.")

    if monthly_returns.empty:
        raise ValueError("Monthly return dataset is empty.")

    if current_month_is_incomplete(as_of_date):
        current_period = as_of_date.to_period("M")
        if (monthly_prices.index.to_period("M") == current_period).any():
            raise ValueError(
                "Monthly price dataset includes the current incomplete calendar month."
            )

        if (monthly_returns.index.to_period("M") == current_period).any():
            raise ValueError(
                "Monthly return dataset includes the current incomplete calendar month."
            )


def build_audit_table(
    daily_prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build ticker-level audit metrics."""
    rows = []
    monthly_missing = monthly_returns.isna().sum()

    for ticker, sector in SECTOR_ETFS.items():
        daily_series = daily_prices[ticker].dropna()
        monthly_return_series = monthly_returns[ticker].dropna()

        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "first_valid_daily_price_date": (
                    daily_series.index.min().date().isoformat()
                    if not daily_series.empty
                    else None
                ),
                "last_valid_daily_price_date": (
                    daily_series.index.max().date().isoformat()
                    if not daily_series.empty
                    else None
                ),
                "valid_daily_observations": int(daily_series.count()),
                "first_valid_monthly_return_date": (
                    monthly_return_series.index.min().date().isoformat()
                    if not monthly_return_series.empty
                    else None
                ),
                "valid_monthly_returns": int(monthly_return_series.count()),
                "monthly_return_missing_values": int(monthly_missing[ticker]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    tickers = list(SECTOR_ETFS)

    downloaded = yf.download(
        tickers=tickers,
        start=START_DATE,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    daily_prices = extract_adjusted_close(downloaded, tickers)
    daily_prices.index = pd.to_datetime(daily_prices.index)
    daily_prices = daily_prices.sort_index()
    daily_prices = daily_prices[~daily_prices.index.duplicated(keep="last")]
    daily_prices = daily_prices.dropna(how="all")

    as_of_date = pd.Timestamp.today().normalize()
    latest_daily_price_date = daily_prices.index.max()

    monthly_prices = daily_prices.resample("ME").last()
    monthly_prices, excluded_incomplete_month = keep_completed_months_only(
        monthly_prices,
        as_of_date,
    )
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    latest_completed_monthly_price_date = monthly_prices.index.max()

    validate_prices(daily_prices, monthly_prices, monthly_returns, tickers, as_of_date)

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MONTHLY_PRICES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    daily_prices.to_csv(RAW_OUTPUT, index_label="Date")
    monthly_prices.to_csv(MONTHLY_PRICES_OUTPUT, index_label="Date")
    monthly_returns.to_csv(MONTHLY_RETURNS_OUTPUT, index_label="Date")

    audit_table = build_audit_table(daily_prices, monthly_returns)

    print("Sector ETF Price Download Audit")
    print("=" * 31)
    print(f"Start date requested: {START_DATE}")
    print(f"Daily price rows: {len(daily_prices)}")
    print(f"Monthly price rows: {len(monthly_prices)}")
    print(f"Monthly return rows: {len(monthly_returns)}")
    print(f"Latest available daily price date: {latest_daily_price_date.date().isoformat()}")
    print(
        "Latest completed monthly price date: "
        f"{latest_completed_monthly_price_date.date().isoformat()}"
    )
    print(
        "Incomplete current-month row excluded: "
        f"{'yes' if excluded_incomplete_month else 'no'}"
    )
    print()
    print(audit_table.to_string(index=False))
    print()
    print("Saved datasets:")
    print(f"- {RAW_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"- {MONTHLY_PRICES_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"- {MONTHLY_RETURNS_OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
