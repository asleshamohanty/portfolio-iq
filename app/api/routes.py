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

# ── Sprint 3 schemas ──────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    weights: dict[str, float]


# ── Sprint 3 endpoints ────────────────────────────────────────────────────────

@router.post("/forecast/volatility")
def forecast_volatility(req: ForecastRequest, db: Session = Depends(get_db)):
    """
    Predict 30-day forward portfolio volatility using XGBoost.
    Returns: prediction, current vol, SHAP top-5 feature drivers,
             and honest walk-forward validation metrics.
    """
    from app.services.ml_engine import train_and_predict
    weights = {k.upper(): v for k, v in req.weights.items()}
    result = train_and_predict(db, weights)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/forecast/explain")
def explain_forecast(req: ForecastRequest, db: Session = Depends(get_db)):
    """
    Plain-English breakdown of why the model predicted what it did.
    Reads SHAP values and returns a human-readable explanation.
    """
    from app.services.ml_engine import train_and_predict
    weights = {k.upper(): v for k, v in req.weights.items()}
    result = train_and_predict(db, weights)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    lines = [
        f"Predicted 30-day volatility: {result['predicted_30d_volatility']*100:.1f}%",
        f"Current 21-day volatility:   {result['current_21d_volatility']*100:.1f}%",
        f"Signal: {result['signal']}",
        "",
        "Top drivers of this prediction:",
    ]
    for i, feat in enumerate(result["shap_top5_features"], 1):
        lines.append(
            f"  {i}. {feat['feature']} — {feat['direction']} "
            f"(SHAP: {feat['shap_value']:+.4f})"
        )

    lines += [
        "",
        f"Validation: MAE={result['validation'].get('mae','N/A')}, "
        f"RMSE={result['validation'].get('rmse','N/A')} "
        f"across {result['validation'].get('n_folds','N/A')} walk-forward folds",
    ]

    return {"explanation": "\n".join(lines), "raw": result}

# ── Sprint 4 schemas ──────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    query: str
    symbol: str = None
    top_k: int = 5


# ── Sprint 4 endpoints ────────────────────────────────────────────────────────

@router.post("/rag/search")
def rag_search(req: RAGRequest, db: Session = Depends(get_db)):
    """
    Semantic search over indexed SEC filings and earnings documents.
    Returns top-K most relevant chunks with similarity scores and sources.
    """
    from app.services.rag_engine import search_documents
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = search_documents(db, req.query, symbol=req.symbol, top_k=req.top_k)

    if not results:
        return {
            "query": req.query,
            "results": [],
            "message": "No relevant documents found. Run document ingestion first.",
        }

    return {
        "query":   req.query,
        "results": results,
        "count":   len(results),
    }