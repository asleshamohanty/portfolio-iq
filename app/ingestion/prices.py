"""
app/ingestion/prices.py

Pulls historical daily OHLCV data from Yahoo Finance for every
ticker in settings.DEFAULT_TICKERS and upserts it into Postgres.

Run standalone: python -m app.ingestion.prices
"""

import time
import yfinance as yf
from datetime import date
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.database import SessionLocal
from app.models.market_data import DailyPrice, Ticker
from app.core.config import settings

# As of yfinance 0.2.51+, curl_cffi browser impersonation is built into
# the library itself (no manual session needed) — yfinance handles the
# cookie/crumb handshake with Yahoo internally. Yahoo's anti-bot system
# is still aggressive though, so the main defense here is pacing:
# real delays between requests, not just retries.
SECTOR_FALLBACK = "Unknown"


def fetch_ticker_info(symbol: str) -> dict:
    """
    Pull basic company metadata (name, sector) for one ticker.
    Best-effort only — if Yahoo's info endpoint is rate-limited or
    returns malformed data, we fall back gracefully instead of
    blocking price ingestion.
    """
    try:
        info = yf.Ticker(symbol).info
        if not info or not isinstance(info, dict):
            raise ValueError("empty or malformed info response")
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", SECTOR_FALLBACK),
        }
    except Exception as e:
        print(f"  ! Could not fetch metadata for {symbol} ({e}) — using fallback")
        return {"name": symbol, "sector": SECTOR_FALLBACK}


def upsert_ticker(db, symbol: str):
    """Insert or refresh ticker metadata row. Always ensures a row exists."""
    existing = db.query(Ticker).filter(Ticker.symbol == symbol).first()
    info = fetch_ticker_info(symbol)
    if existing:
        existing.name = info["name"]
        existing.sector = info["sector"]
    else:
        db.add(Ticker(symbol=symbol, name=info["name"], sector=info["sector"]))
    db.commit()


def fetch_and_store_prices(symbol: str, period: str = "5y", retries: int = 3):
    """
    Download `period` worth of daily prices for `symbol` and
    upsert each row into daily_prices. Upsert means: if a row for
    that symbol+date already exists, update it instead of erroring.

    Retries with backoff since Yahoo intermittently rate-limits —
    a transient failure shouldn't mean zero data for that ticker.
    """
    print(f"Fetching {symbol} ({period})...")

    df = None
    for attempt in range(1, retries + 1):
        try:
            # auto_adjust=True (default) gives split/dividend-adjusted close
            df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
            if not df.empty:
                break
            print(f"  ! Empty response on attempt {attempt}/{retries}, retrying...")
        except Exception as e:
            print(f"  ! Error on attempt {attempt}/{retries}: {e}")
        time.sleep(8 * attempt)  # backoff: 8s, 16s, 24s — Yahoo 429s need real cooldown

    if df is None or df.empty:
        print(f"  ! No data returned for {symbol} after {retries} attempts — skipping")
        return 0

    db = SessionLocal()
    upsert_ticker(db, symbol)

    rows_written = 0
    for idx, row in df.iterrows():
        stmt = pg_insert(DailyPrice).values(
            symbol=symbol,
            date=idx.date(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            adj_close=float(row.get("Close", row["Close"])),  # yfinance auto-adjusts by default
            volume=float(row["Volume"]),
        )
        # ON CONFLICT (symbol, date) DO UPDATE — this is the "upsert"
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "adj_close": stmt.excluded.adj_close,
                "volume": stmt.excluded.volume,
            },
        )
        db.execute(stmt)
        rows_written += 1

    db.commit()
    db.close()
    print(f"  -> {rows_written} rows written for {symbol}")
    return rows_written


def run_all():
    """Ingest price history for every ticker in the watchlist."""
    total = 0
    for i, symbol in enumerate(settings.ticker_list):
        total += fetch_and_store_prices(symbol)
        if i < len(settings.ticker_list) - 1:
            time.sleep(6)  # be polite to Yahoo's servers between tickers
    print(f"\nDone. {total} total rows processed across {len(settings.ticker_list)} tickers.")


if __name__ == "__main__":
    run_all()
