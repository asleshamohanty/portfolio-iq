"""
app/api/routes.py

Sprint 1 endpoints: /health, /tickers, /prices/{symbol}, /macro/{series_id}
Sprint 2 endpoints: /portfolio (POST), /risk/analyse (POST), /risk/stress-test (POST)
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel, field_validator

from app.db.database import get_db
from app.models.market_data import Ticker, DailyPrice, MacroIndicator
from app.models.portfolio import Portfolio, Holding
from app.services.quant_engine import full_risk_report, stress_test

router = APIRouter()


# ── Sprint 1 endpoints ────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/tickers")
def list_tickers(db: Session = Depends(get_db)):
    tickers = db.query(Ticker).all()
    return {
        "data": [{"symbol": t.symbol, "name": t.name, "sector": t.sector} for t in tickers],
        "count": len(tickers),
    }


@router.get("/prices/{symbol}")
def get_prices(
    symbol: str,
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    limit: int = Query(100, le=2000),
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    q = db.query(DailyPrice).filter(DailyPrice.symbol == symbol)
    if start:
        q = q.filter(DailyPrice.date >= start)
    if end:
        q = q.filter(DailyPrice.date <= end)
    rows = q.order_by(desc(DailyPrice.date)).limit(limit).all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data for '{symbol}'")
    return {
        "symbol": symbol,
        "data": [{"date": r.date.isoformat(), "open": r.open, "high": r.high,
                  "low": r.low, "close": r.close, "volume": r.volume} for r in rows],
        "count": len(rows),
    }


@router.get("/macro/{series_id}")
def get_macro_series(
    series_id: str,
    limit: int = Query(100, le=2000),
    db: Session = Depends(get_db),
):
    series_id = series_id.upper()
    rows = (db.query(MacroIndicator)
            .filter(MacroIndicator.series_id == series_id)
            .order_by(desc(MacroIndicator.date)).limit(limit).all())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for series '{series_id}'")
    return {
        "series_id": series_id,
        "series_name": rows[0].series_name,
        "data": [{"date": r.date.isoformat(), "value": r.value} for r in rows],
        "count": len(rows),
    }


# ── Sprint 2 schemas ──────────────────────────────────────────────────────────

class PortfolioRequest(BaseModel):
    name: str
    weights: dict[str, float]  # e.g. {"AAPL": 0.4, "MSFT": 0.3, "GS": 0.3}

    @field_validator("weights")
    @classmethod
    def weights_must_sum_to_one(cls, v):
        total = sum(v.values())
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"Weights must sum to ~1.0, got {total:.4f}")
        return v


class RiskRequest(BaseModel):
    weights: dict[str, float]
    lookback_days: int = 252  # default 1 year


class StressRequest(BaseModel):
    weights: dict[str, float]


# ── Sprint 2 endpoints ────────────────────────────────────────────────────────

@router.post("/portfolio")
def create_portfolio(req: PortfolioRequest, db: Session = Depends(get_db)):
    """
    Save a named portfolio with ticker weights.
    Weights must sum to 1.0 (±2% tolerance).
    """
    portfolio = Portfolio(name=req.name)
    db.add(portfolio)
    db.flush()  # get the id without committing

    for symbol, weight in req.weights.items():
        db.add(Holding(portfolio_id=portfolio.id, symbol=symbol.upper(), weight=weight))

    db.commit()
    return {
        "portfolio_id": portfolio.id,
        "name": portfolio.name,
        "holdings": req.weights,
        "message": "Portfolio saved. Run /risk/analyse to get a risk report.",
    }


@router.post("/risk/analyse")
def analyse_risk(req: RiskRequest, db: Session = Depends(get_db)):
    """
    Full risk report for any portfolio (pass weights directly, no need to save first).
    Returns: volatility, Sharpe, Sortino, max drawdown, VaR, CVaR, sector exposure,
             correlation matrix, and stress test results all in one response.
    """
    weights = {k.upper(): v for k, v in req.weights.items()}
    report = full_risk_report(db, weights, req.lookback_days)
    if "error" in report:
        raise HTTPException(status_code=400, detail=report["error"])
    return report


@router.post("/risk/stress-test")
def run_stress_test(req: StressRequest, db: Session = Depends(get_db)):
    """
    Run portfolio through 2008 GFC, 2020 COVID crash, and 2022 rate hike cycle.
    Shows cumulative return, max drawdown, and annualised volatility per scenario.
    """
    weights = {k.upper(): v for k, v in req.weights.items()}
    return {"weights": weights, "scenarios": stress_test(db, weights)}