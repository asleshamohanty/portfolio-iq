"""
app/ingestion/macro.py

Pulls macroeconomic series from FRED (Federal Reserve Economic Data)
and upserts into Postgres. Get a free API key at:
https://fred.stlouisfed.org/docs/api/api_key.html

Run standalone: python -m app.ingestion.macro
"""

from fredapi import Fred
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.database import SessionLocal
from app.models.market_data import MacroIndicator
from app.core.config import settings

# Series we care about for portfolio risk context.
# Key = FRED series ID, Value = human-readable name
SERIES = {
    "DFF": "Federal Funds Effective Rate",
    "CPIAUCSL": "Consumer Price Index (CPI)",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "GDP": "Gross Domestic Product",
    "UNRATE": "Unemployment Rate",
}


def fetch_and_store_series(fred: Fred, series_id: str, name: str):
    print(f"Fetching {series_id} ({name})...")
    try:
        data = fred.get_series(series_id)
    except Exception as e:
        print(f"  ! Failed to fetch {series_id}: {e}")
        return 0

    db = SessionLocal()
    rows_written = 0
    for dt, value in data.items():
        if value is None or (isinstance(value, float) and value != value):  # NaN check
            continue
        stmt = pg_insert(MacroIndicator).values(
            series_id=series_id,
            series_name=name,
            date=dt.date(),
            value=float(value),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["series_id", "date"],
            set_={"value": stmt.excluded.value, "series_name": stmt.excluded.series_name},
        )
        db.execute(stmt)
        rows_written += 1

    db.commit()
    db.close()
    print(f"  -> {rows_written} rows written for {series_id}")
    return rows_written


def run_all():
    if not settings.FRED_API_KEY:
        print("! FRED_API_KEY not set in .env — skipping macro ingestion.")
        print("  Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    fred = Fred(api_key=settings.FRED_API_KEY)
    total = 0
    for series_id, name in SERIES.items():
        total += fetch_and_store_series(fred, series_id, name)
    print(f"\nDone. {total} total rows processed across {len(SERIES)} series.")


if __name__ == "__main__":
    run_all()
