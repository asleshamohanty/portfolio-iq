"""
app/services/quant_engine.py

The quantitative analytics engine — Layer 3 of the architecture.
Pure finance math. No ML, no LLM. Every number here is auditable.

Computes:
  - Annualised volatility
  - Sharpe ratio / Sortino ratio
  - Max drawdown
  - Value at Risk (historical simulation)
  - CVaR (Expected Shortfall)
  - Correlation matrix
  - Sector exposure
  - Historical stress tests (2008, 2020, 2022)
"""

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from app.models.market_data import DailyPrice, Ticker


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_prices(db: Session, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Pull adjusted close prices for given symbols between two dates.
    Returns a DataFrame: index=date, columns=symbols.
    """
    rows = (
        db.query(DailyPrice.date, DailyPrice.symbol, DailyPrice.close)
        .filter(DailyPrice.symbol.in_(symbols))
        .filter(DailyPrice.date >= start)
        .filter(DailyPrice.date <= end)
        .order_by(DailyPrice.date)
        .all()
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    return df.pivot(index="date", columns="symbol", values="close").dropna()


def _portfolio_returns(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Daily portfolio returns = weighted sum of each stock's daily return.
    weights should sum to 1.0.
    """
    symbols = [s for s in weights if s in prices.columns]
    w = np.array([weights[s] for s in symbols])
    w = w / w.sum()  # normalise in case weights don't perfectly sum to 1
    returns = prices[symbols].pct_change().dropna()
    return returns.dot(w)


# ── Core metrics ──────────────────────────────────────────────────────────────

def annualised_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns × sqrt(252 trading days)."""
    return float(returns.std() * np.sqrt(252))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    (Annualised return - risk free rate) / annualised volatility.
    Risk free rate defaults to 5% (approximate current Fed Funds).
    """
    ann_return = float(returns.mean() * 252)
    ann_vol = annualised_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Like Sharpe but only penalises downside volatility.
    Preferred by many risk managers over Sharpe.
    """
    ann_return = float(returns.mean() * 252)
    downside = returns[returns < 0].std() * np.sqrt(252)
    if downside == 0:
        return 0.0
    return (ann_return - risk_free_rate) / float(downside)


def max_drawdown(returns: pd.Series) -> float:
    """
    Largest peak-to-trough decline in the portfolio's value.
    Expressed as a negative percentage.
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical simulation VaR.
    The loss you'd expect not to exceed on 95% of days.
    Negative number = a loss.
    """
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall.
    Average of all returns WORSE than the VaR threshold.
    A more complete picture of tail risk than VaR alone.
    """
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if not tail.empty else var


def correlation_matrix(prices: pd.DataFrame, symbols: list[str]) -> dict:
    """Pairwise correlation between all holdings. Returns a nested dict."""
    available = [s for s in symbols if s in prices.columns]
    returns = prices[available].pct_change().dropna()
    corr = returns.corr().round(4)
    return corr.to_dict()


def sector_exposure(db: Session, weights: dict[str, float]) -> dict[str, float]:
    """
    Group portfolio weight by sector using the tickers table.
    Returns e.g. {"Technology": 0.45, "Financial Services": 0.30}
    """
    exposure: dict[str, float] = {}
    for symbol, weight in weights.items():
        ticker = db.query(Ticker).filter(Ticker.symbol == symbol).first()
        sector = ticker.sector if ticker and ticker.sector else "Unknown"
        exposure[sector] = exposure.get(sector, 0) + weight
    return {k: round(v, 4) for k, v in exposure.items()}


# ── Stress tests ──────────────────────────────────────────────────────────────

STRESS_PERIODS = {
    "2008_financial_crisis": ("2008-09-01", "2009-03-31"),
    "2020_covid_crash":      ("2020-02-01", "2020-04-30"),
    "2022_rate_hike_cycle":  ("2022-01-01", "2022-12-31"),
}


def stress_test(db: Session, weights: dict[str, float]) -> dict:
    """
    Run the portfolio through each historical crisis period.
    Shows how your current holdings would have performed.
    """
    results = {}
    symbols = list(weights.keys())

    for scenario, (start, end) in STRESS_PERIODS.items():
        prices = _load_prices(db, symbols, start, end)
        if prices.empty or len(prices) < 5:
            results[scenario] = {"error": "insufficient data for this period"}
            continue

        port_returns = _portfolio_returns(prices, weights)
        cumulative_return = float((1 + port_returns).prod() - 1)
        results[scenario] = {
            "period": f"{start} to {end}",
            "cumulative_return": round(cumulative_return, 4),
            "max_drawdown": round(max_drawdown(port_returns), 4),
            "volatility": round(annualised_volatility(port_returns), 4),
        }

    return results


# ── Main analysis function ────────────────────────────────────────────────────

def full_risk_report(db: Session, weights: dict[str, float], lookback_days: int = 252) -> dict:
    """
    The full Sprint 2 risk report.
    Called by POST /risk/analyse.
    """
    from datetime import date, timedelta
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days + 30)).isoformat()

    symbols = list(weights.keys())
    prices = _load_prices(db, symbols, start, end)

    if prices.empty:
        return {"error": "No price data found. Run ingestion first."}

    port_returns = _portfolio_returns(prices, weights)

    return {
        "portfolio": weights,
        "lookback_days": lookback_days,
        "metrics": {
            "annualised_volatility": round(annualised_volatility(port_returns), 4),
            "sharpe_ratio":          round(sharpe_ratio(port_returns), 4),
            "sortino_ratio":         round(sortino_ratio(port_returns), 4),
            "max_drawdown":          round(max_drawdown(port_returns), 4),
            "var_95":                round(value_at_risk(port_returns, 0.95), 4),
            "cvar_95":               round(conditional_var(port_returns, 0.95), 4),
        },
        "sector_exposure":  sector_exposure(db, weights),
        "correlation":      correlation_matrix(prices, symbols),
        "stress_tests":     stress_test(db, weights),
    }