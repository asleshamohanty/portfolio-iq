"""
app/api/routes.py

Sprint 1 endpoints. Just enough to prove the data layer works:
  GET /health                 — is the API alive?
  GET /tickers                — list everything we're tracking
  GET /prices/{symbol}        — price history for one ticker
  GET /macro/{series_id}      — one macro series over time
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database import get_db
from app.models.market_data import Ticker, DailyPrice, MacroIndicator

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/tickers")
def list_tickers(db: Session = Depends(get_db)):
    """Return every ticker currently tracked in the database."""
    tickers = db.query(Ticker).all()
    return {
        "data": [
            {"symbol": t.symbol, "name": t.name, "sector": t.sector}
            for t in tickers
        ],
        "count": len(tickers),
    }


@router.get("/prices/{symbol}")
def get_prices(
    symbol: str,
    start: Optional[date] = Query(None, description="Filter from this date (YYYY-MM-DD)"),
    end: Optional[date] = Query(None, description="Filter to this date (YYYY-MM-DD)"),
    limit: int = Query(100, le=2000),
    db: Session = Depends(get_db),
):
    """Return daily price history for one ticker, most recent first."""
    symbol = symbol.upper()
    q = db.query(DailyPrice).filter(DailyPrice.symbol == symbol)

    if start:
        q = q.filter(DailyPrice.date >= start)
    if end:
        q = q.filter(DailyPrice.date <= end)

    rows = q.order_by(desc(DailyPrice.date)).limit(limit).all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No price data found for '{symbol}'. Has ingestion been run for this ticker?",
        )

    return {
        "symbol": symbol,
        "data": [
            {
                "date": r.date.isoformat(),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/macro/{series_id}")
def get_macro_series(
    series_id: str,
    limit: int = Query(100, le=2000),
    db: Session = Depends(get_db),
):
    """Return one macroeconomic series over time, most recent first."""
    series_id = series_id.upper()
    rows = (
        db.query(MacroIndicator)
        .filter(MacroIndicator.series_id == series_id)
        .order_by(desc(MacroIndicator.date))
        .limit(limit)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for series '{series_id}'. Has macro ingestion been run?",
        )

    return {
        "series_id": series_id,
        "series_name": rows[0].series_name,
        "data": [{"date": r.date.isoformat(), "value": r.value} for r in rows],
        "count": len(rows),
    }
