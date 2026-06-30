"""
app/models/market_data.py

Three tables for Sprint 1:

  tickers           — the universe of stocks we're tracking
  daily_prices      — OHLCV price history per ticker per day
  macro_indicators  — FRED economic series (rates, CPI, VIX, etc.)

Kept deliberately simple in Sprint 1. Sprint 2 adds a `portfolios`
table and a `features` table on top of this foundation.
"""

from sqlalchemy import Column, Integer, String, Float, Date, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy import DateTime

from app.db.database import Base


class Ticker(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255))
    sector = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DailyPrice(Base):
    __tablename__ = "daily_prices"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        # Never store the same ticker+date twice — re-running ingestion
        # should update, not duplicate.
        UniqueConstraint("symbol", "date", name="uq_symbol_date"),
    )


class MacroIndicator(Base):
    __tablename__ = "macro_indicators"

    id = Column(Integer, primary_key=True)
    series_id = Column(String(20), nullable=False, index=True)  # e.g. "DFF", "CPIAUCSL"
    series_name = Column(String(255))
    date = Column(Date, nullable=False, index=True)
    value = Column(Float)

    __table_args__ = (
        UniqueConstraint("series_id", "date", name="uq_series_date"),
    )
