# PortfolioIQ

AI-powered portfolio intelligence platform — quantitative analytics, ML forecasting, RAG-grounded document retrieval, and LLM reasoning behind a single FastAPI interface. Architecture inspired by BlackRock Aladdin and Goldman Sachs Marquee.

**This repo currently implements Sprint 1: Data Foundation — fully built, containerized, and verified end to end with real market data.** See the full 7-layer architecture and 6-sprint roadmap in `PortfolioIQ_Blueprint.pdf`.

## Sprint 1 — What's Working

- PostgreSQL schema: `tickers`, `daily_prices`, `macro_indicators`
- Price ingestion from Yahoo Finance — verified pulling 5 years of daily OHLCV (1,254 trading days) for a 10-stock watchlist spanning tech, financials, and healthcare
- Macro ingestion from FRED — verified pulling 5 economic series (37,720+ data points): Fed Funds Rate, CPI, VIX, GDP, Unemployment Rate
- Automatic sector classification per ticker (e.g. GS and BLK both correctly tagged "Financial Services")
- Idempotent ingestion — safe to re-run; uses Postgres upserts so re-running never creates duplicate rows
- FastAPI serving live data with auto-generated Swagger docs at `/docs`
- Fully containerized with Docker Compose — one command brings up Postgres + API together

## Quick Start

### 1. Prerequisites

- Docker Desktop installed and running
- A free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html (optional — price data works without it, only macro ingestion needs it)

### 2. Setup

\`\`\`bash
# Clone and enter the project
git clone https://github.com/YOUR_USERNAME/portfolioiq.git
cd portfolioiq

# Copy the env template and fill in your FRED key
cp .env.example .env
# then edit .env and paste your FRED_API_KEY

# Build and start both containers (Postgres + API)
docker-compose up --build
\`\`\`

Leave that running in one terminal. The API is now live at `http://localhost:8000` and Postgres is live at `localhost:5432`.

### 3. Load data

In a **second terminal**, with the containers still running:

\`\`\`bash
docker-compose exec api python scripts/run_ingestion.py
\`\`\`

This pulls 5 years of price history for AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, GS, BLK, V, JNJ, plus 5 macro series from FRED. Takes 1-2 minutes.

### 4. Try it

Open `http://localhost:8000/docs` in your browser for interactive Swagger UI, or use curl:

\`\`\`bash
curl http://localhost:8000/tickers
curl http://localhost:8000/prices/AAPL
curl http://localhost:8000/prices/GS?limit=10
curl http://localhost:8000/macro/VIXCLS
\`\`\`

### 5. Run tests

\`\`\`bash
docker-compose exec api pytest
\`\`\`

## A Note on Yahoo Finance Rate Limiting

Yahoo Finance's API has tightened anti-bot defenses and will return `429 Too Many Requests` if hit too quickly. The ingestion script paces requests deliberately (built-in delays between tickers, retries with backoff) specifically to work around this reliably rather than failing silently. If you ever see rate-limit errors, wait 15-20 minutes before retrying — this is a Yahoo-side limit, not a bug in this codebase.

## Project Structure

\`\`\`
portfolioiq/
├── app/
│   ├── core/config.py        # Settings, env vars
│   ├── db/database.py        # SQLAlchemy engine/session
│   ├── models/market_data.py # Table definitions
│   ├── ingestion/
│   │   ├── prices.py         # yfinance puller (rate-limit aware, retry logic)
│   │   └── macro.py          # FRED puller
│   ├── api/routes.py         # Endpoint handlers
│   └── main.py                # FastAPI entrypoint
├── scripts/run_ingestion.py  # Manual full data pull
├── tests/test_api.py
├── docker-compose.yml         # Postgres + API
├── Dockerfile
└── requirements.txt
\`\`\`

## Roadmap

This is Sprint 1 of 6. Full architecture (7 layers), API reference, and sprint-by-sprint plan are documented in `PortfolioIQ_Blueprint.pdf`.

- **Sprint 1** ✅ Data Foundation — *this repo, verified working*
- **Sprint 2** — Quantitative Analytics Engine (volatility, VaR, CVaR, Sharpe, correlation matrix, historical stress testing — the mini Aladdin/Marquee risk core)
- **Sprint 3** — ML Forecasting (XGBoost 30-day volatility prediction, walk-forward validation, SHAP explainability)
- **Sprint 4** — RAG Engine (SEC filings + earnings transcripts indexed via pgvector, narrow-and-deep over 15 companies)
- **Sprint 5** — LLM Reasoning Layer (`/chat`, `/chat/stress-test` — the flagship endpoint)
- **Sprint 6** — Cloud Deployment (AWS, CI/CD, public live demo)

## Tech Stack

Python, FastAPI, PostgreSQL, SQLAlchemy, Docker, yfinance, fredapi

## Design Philosophy

The LLM never calculates — models and quant formulas do. The LLM's job (added in Sprint 5) is to synthesize structured outputs from the analytics, ML, and retrieval layers into clear, grounded explanations. This separation is what distinguishes this architecture from a generic finance chatbot, and mirrors how production systems like Aladdin and Marquee are actually built.