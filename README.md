# PortfolioIQ

AI-powered portfolio intelligence platform combining quantitative analytics, ML forecasting, RAG-grounded document retrieval, and LLM reasoning behind a single FastAPI interface.

The architecture is inspired by BlackRock Aladdin and Goldman Sachs Marquee.

> **Current Status:** Sprint 1 (Data Foundation) is fully implemented, containerized, and verified end-to-end using real market data.

The complete 7-layer architecture and 6-sprint roadmap are documented in `PortfolioIQ_Blueprint.pdf`.

---

# Sprint 1 — What's Working

- PostgreSQL schema:
  - `tickers`
  - `daily_prices`
  - `macro_indicators`
- Price ingestion from Yahoo Finance
  - Verified pulling **5 years** of daily OHLCV data (**1,254 trading days**) for a 10-stock watchlist across technology, financials, and healthcare.
- Macroeconomic ingestion from FRED
  - Verified pulling **5 economic series** (**37,720+ observations**):
    - Fed Funds Rate
    - CPI
    - VIX
    - GDP
    - Unemployment Rate
- Automatic sector classification for each ticker (e.g. GS and BLK correctly classified as **Financial Services**)
- Idempotent ingestion using PostgreSQL upserts (safe to rerun without creating duplicates)
- FastAPI serving live market data with auto-generated Swagger documentation at `/docs`
- Fully containerized using Docker Compose

---

# Quick Start

## 1. Prerequisites

- Docker Desktop installed and running
- A free FRED API key (optional; only required for macroeconomic data)

https://fred.stlouisfed.org/docs/api/api_key.html

---

## 2. Setup

```bash
# Clone the repository
git clone https://github.com/asleshamohanty/portfolioiq.git

cd portfolioiq

# Copy environment variables
cp .env.example .env

# Edit .env and add your FRED_API_KEY

# Build and start the API + PostgreSQL
docker compose up --build
```

Leave Docker running.

The application will be available at:

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- PostgreSQL: localhost:5432

---

## 3. Load Market Data

Open a **second terminal** while Docker is still running.

```bash
docker compose exec api python scripts/run_ingestion.py
```

This downloads:

- 5 years of daily prices for:

```
AAPL
MSFT
GOOGL
AMZN
NVDA
JPM
GS
BLK
V
JNJ
```

and imports these FRED macroeconomic series:

- Fed Funds Rate
- CPI
- VIX
- GDP
- Unemployment Rate

Typical runtime: **1–2 minutes**

---

## 4. Explore the API

Swagger UI:

```
http://localhost:8000/docs
```

Example requests:

```bash
curl http://localhost:8000/tickers

curl http://localhost:8000/prices/AAPL

curl "http://localhost:8000/prices/GS?limit=10"

curl http://localhost:8000/macro/VIXCLS
```

---

## 5. Run Tests

```bash
docker compose exec api pytest
```

---

# Yahoo Finance Rate Limiting

Yahoo Finance has strengthened its anti-bot protections and may return:

```
429 Too Many Requests
```

The ingestion pipeline is designed to handle this by using:

- request pacing
- retry logic
- exponential backoff

If rate limiting still occurs, wait approximately **15–20 minutes** before retrying. This is a limitation imposed by Yahoo Finance rather than an issue with this project.

---

# Project Structure

```text
portfolioiq/
├── app/
│   ├── core/
│   │   └── config.py
│   ├── db/
│   │   └── database.py
│   ├── models/
│   │   └── market_data.py
│   ├── ingestion/
│   │   ├── prices.py
│   │   └── macro.py
│   ├── api/
│   │   └── routes.py
│   └── main.py
├── scripts/
│   └── run_ingestion.py
├── tests/
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

# Roadmap

This repository currently implements **Sprint 1**.

The complete system architecture, API specification, and development roadmap are documented in `PortfolioIQ_Blueprint.pdf`.

| Sprint | Status | Description |
|---------|--------|-------------|
| Sprint 1 | Completed | Data Foundation |
| Sprint 2 | Planned | Quantitative Analytics Engine (volatility, VaR, CVaR, Sharpe ratio, correlation matrix, historical stress testing) |
| Sprint 3 | Planned | ML Forecasting (XGBoost volatility prediction, walk-forward validation, SHAP explainability) |
| Sprint 4 | Planned | RAG Engine (SEC filings + earnings transcripts using pgvector) |
| Sprint 5 | Planned | LLM Reasoning Layer (`/chat`, `/chat/stress-test`) |
| Sprint 6 | Planned | Cloud Deployment (AWS, CI/CD, public demo) |

---

# Tech Stack

- Python
- FastAPI
- PostgreSQL
- SQLAlchemy
- Docker
- yfinance
- fredapi

---

# Design Philosophy

PortfolioIQ separates computation from reasoning.

The quantitative analytics, machine learning models, and retrieval systems perform all calculations.

The LLM (introduced in Sprint 5) is responsible only for synthesizing structured outputs from those systems into grounded, explainable insights.

This separation mirrors the architecture of production-grade financial platforms such as BlackRock Aladdin and Goldman Sachs Marquee, rather than relying on an LLM to perform financial calculations directly.