# PortfolioIQ

> An AI-powered portfolio intelligence platform that combines quantitative risk analytics, machine learning forecasting, document retrieval, and large language model reasoning behind a single API interface.

Built as a demonstration of how modern financial AI systems are architected — not as a chatbot wrapper around market data, but as a layered intelligence platform where every answer is grounded in mathematics, validated models, and primary source documents.

---

## The Problem

Financial intelligence is fragmented. A portfolio manager today consults a risk dashboard for volatility metrics, a Bloomberg terminal for market context, Excel models for stress testing, research reports for qualitative insight, and news feeds for real-time signals — switching between tools constantly, synthesising manually before every decision.

The result is slow, error-prone, and impossible to scale. The insight exists. The bottleneck is integration.

---

## The Solution

PortfolioIQ unifies quantitative analytics, predictive machine learning, document retrieval, and language model reasoning behind a single API. One request returns a complete picture: what the numbers say, what the models predict, what the filings confirm, and what it means in plain English.

The key architectural decision: **the LLM never calculates**. Every number in every response is produced by the quant engine or the ML model. The LLM's only job is to synthesise those outputs into clear, grounded prose — the way a senior analyst would brief a portfolio manager. This separation is what makes the system trustworthy. If the LLM hallucinates a number, the system catches it because the number wasn't in the context it received.

---

## Features

**Quantitative Risk Engine**
Computes annualised volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Value at Risk (historical simulation), Conditional VaR, pairwise correlation matrix, and sector concentration. All calculations are mathematically transparent and auditable. Historical stress tests run the portfolio through the 2008 financial crisis, the 2020 COVID crash, and the 2022 rate hike cycle using actual price data.

**Machine Learning Forecasting**
XGBoost regression model predicts 30-day forward portfolio volatility using 19 engineered features spanning rolling returns, momentum signals, volatility regimes, and macroeconomic indicators. Validated using strict walk-forward cross-validation — no future data leakage, no inflated metrics. SHAP values explain every prediction, identifying which features drove the forecast.

**RAG Document Intelligence**
1,802 chunks from 10-K SEC filings across 10 major companies, embedded using `all-MiniLM-L6-v2` and stored in pgvector. At query time, the user's question is embedded and matched against the corpus via cosine similarity search directly in PostgreSQL. A relevance threshold filters low-quality retrievals before they reach the LLM — the system will explicitly say when it cannot find supporting evidence rather than fabricate it.

**LLM Reasoning Layer**
Gemini 2.5 Flash receives a structured context containing computed risk metrics, ML predictions, and retrieved document chunks. It produces institutional-grade prose — the kind of analysis a Managing Director would deliver to a client. No bullet points, no markdown, no hedging. Grounded claims, specific numbers, source citations.

**Flagship Endpoint: `/chat/stress-test`**
Pass a plain-English scenario and a portfolio. The system maps the scenario to historical analogue periods, runs the quant engine through that period's price data, retrieves relevant regulatory filings and risk disclosures, and returns a four-paragraph institutional risk report: scenario interpretation, portfolio impact with real numbers, sector vulnerabilities, and actionable recommendations.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Data Sources                      │
│  Yahoo Finance · FRED · SEC EDGAR · Earnings Calls  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                 Ingestion Layer                      │
│        Price Pipeline · Macro Pipeline · Doc Pipeline│
└────────────┬─────────────────────────┬──────────────┘
             │                         │
             ▼                         ▼
    ┌─────────────────┐      ┌──────────────────┐
    │   PostgreSQL    │      │    pgvector       │
    │ prices · macro  │      │  doc embeddings  │
    │ portfolios · logs│      │  cosine search   │
    └────────┬────────┘      └────────┬─────────┘
             │                         │
             ▼                         ▼
┌────────────────────────────────────────────────────┐
│                   Engine Layer                      │
│  Quant Engine · ML Engine (XGBoost) · RAG Engine   │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│           Financial Reasoning Layer                 │
│              LLM (Gemini 2.5 Flash)                │
│   metrics + predictions + documents → prose report │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│                FastAPI Gateway                      │
│  /risk/analyse · /forecast/volatility · /chat      │
│  /rag/search · /chat/stress-test · /portfolio      │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
                      User
```

**Layer 1 — Data Ingestion**
yfinance pulls 5 years of daily OHLCV for 10 tickers with retry logic and rate-limit handling. fredapi pulls 5 macroeconomic series from the Federal Reserve. SEC EDGAR's free API fetches 10-K filings for 10 companies. All ingestion is idempotent — PostgreSQL upserts mean re-running the pipeline never creates duplicates.

**Layer 2 — Feature Engineering**
Daily and rolling returns, volatility regimes, momentum signals, rolling Sharpe, drawdown windows, and macroeconomic rate-of-change features. These feed both the quant engine and the ML model.

**Layer 3 — Quantitative Analytics Engine**
Pure mathematics. No models, no predictions. Volatility, correlation, VaR, CVaR, sector exposure, and stress tests computed directly from historical price data. This is the trust anchor of the system — every output is reproducible from first principles.

**Layer 4 — Machine Learning Engine**
XGBoost trained on the feature matrix with walk-forward validation. SHAP explainability on every prediction. One model, done properly, rather than four models done poorly.

**Layer 5 — RAG Engine**
Documents chunked at 500 tokens with 50-token overlap. Embedded with sentence-transformers. Stored in pgvector. Retrieved by cosine similarity at query time. Relevance threshold of 0.30 — below this the chunk is discarded rather than passed to the LLM.

**Layer 6 — LLM Reasoning Layer**
Receives structured context: metrics first, predictions second, documents third, question last. This ordering is deliberate — it anchors the LLM in quantitative reality before it sees the qualitative question. System prompt instructs institutional prose, no fabrication, explicit uncertainty when evidence is absent.

**Layer 7 — FastAPI Gateway**
Auto-generated OpenAPI docs at `/docs`. Pydantic validation on every request. Background tasks for heavy computations. JWT-ready middleware. Consistent JSON response schema across all endpoints.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API status |
| GET | `/tickers` | All tracked companies with sector tags |
| GET | `/prices/{symbol}` | Price history with date range filters |
| GET | `/macro/{series_id}` | Macroeconomic series data |
| POST | `/portfolio` | Save a named portfolio with weights |
| POST | `/risk/analyse` | Full risk report: volatility, Sharpe, VaR, CVaR, drawdown, correlation, sector exposure, stress tests |
| POST | `/risk/stress-test` | Historical stress scenarios: 2008, 2020, 2022 |
| POST | `/forecast/volatility` | 30-day volatility prediction with SHAP explanation |
| POST | `/forecast/explain` | Plain-English breakdown of the ML forecast |
| POST | `/rag/search` | Semantic search over SEC filings |
| POST | `/chat` | Natural language Q&A grounded in live analytics + documents |
| POST | `/chat/stress-test` | **Flagship.** Plain-English scenario → institutional risk report |

---

## Design Principles

**The LLM synthesises. Models calculate.**
This is the most important architectural decision in the system. Language models are powerful synthesisers but unreliable calculators. By separating calculation (quant engine, ML model) from synthesis (LLM), each component does what it is actually good at. The LLM never sees raw data — only pre-computed, structured outputs. This makes every response auditable: if a number appears in the LLM's answer, that number exists in the context it received.

**Quantitative math is the trust anchor.**
Risk metrics computed by the quant engine are mathematically deterministic. Given the same price data, the same inputs will always produce the same outputs. This is not true of language models. By grounding every LLM response in quant outputs, the system inherits the reliability of the mathematics.

**Retrieval quality over retrieval volume.**
The RAG engine indexes 1,802 chunks across 10 companies — narrow and deep rather than broad and shallow. A 0.30 cosine similarity threshold means low-relevance chunks are discarded before reaching the LLM. The system will explicitly acknowledge when it cannot find supporting evidence rather than pass irrelevant context and hope the LLM figures it out.

**Walk-forward validation, not train-test split.**
The ML model is validated using strict chronological folds — the model never sees future data during training. Many finance ML projects use random train-test splits which leak future information and produce fraudulently optimistic metrics. Walk-forward validation produces honest numbers that reflect real-world performance.

**Idempotent data pipelines.**
All ingestion uses PostgreSQL upserts. Re-running any ingestion script on any day produces the same database state — no duplicates, no corruption. This is a production-grade requirement that many student projects skip.

**One interface, many consumers.**
Every intelligence component is exposed through FastAPI. The dashboard, a future mobile client, a third-party integration — all consume the same endpoints. This mirrors the API-first architecture of platforms like Marquee and Aladdin, where data and analytics are made programmable rather than buried in internal systems.

---

## Data Sources

| Source | What It Provides | Access |
|--------|-----------------|--------|
| Yahoo Finance | 5 years daily OHLCV for 10 tickers | yfinance library, free |
| FRED (Federal Reserve) | Fed Funds Rate, CPI, VIX, GDP, Unemployment | fredapi + free API key |
| SEC EDGAR | 10-K annual filings for 10 companies | EDGAR REST API, no auth required |
| pgvector | Embedding storage and cosine similarity search | PostgreSQL extension, no separate DB |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API Framework | FastAPI + Pydantic | Auto-generated docs, type safety, async support |
| Database | PostgreSQL + SQLAlchemy | Relational integrity, time series, audit logs |
| Vector Search | pgvector | Embedding storage without a separate vector database |
| Market Data | yfinance | Free, reliable, 50+ years of price history |
| Macro Data | fredapi | Official Federal Reserve data, 800k+ series |
| Document Source | SEC EDGAR API | Free, official, no authentication required |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, strong on financial text |
| LLM | Gemini 2.5 Flash | Fast, cost-efficient, strong reasoning |
| ML | XGBoost + SHAP | Industry standard, interpretable, fast |
| Containerisation | Docker + Docker Compose | One-command reproducible environment |

---

## Verified Results

Every sprint produced a working, deployable version of the platform.

**Sprint 1 — Data Foundation**
- ✅ 12,540 price rows across 10 tickers (1,254 trading days each)
- ✅ 37,720 macroeconomic data points across 5 FRED series
- ✅ Sector classification verified (GS → Financial Services, AAPL → Technology)
- ✅ Idempotent ingestion confirmed — re-running produces no duplicates

**Sprint 2 — Quantitative Analytics Engine**
- ✅ Full risk report: 15.66% annualised volatility, Sharpe 0.30, Sortino 0.44
- ✅ Max drawdown -10.75%, VaR 95% -1.73%, CVaR -2.26%
- ✅ Sector exposure: Technology 50%, Financial Services 35%, Healthcare 15%
- ✅ 2022 stress test: -17.85% cumulative return, -24.69% max drawdown

**Sprint 3 — ML Forecasting**
- ✅ XGBoost predicting 30-day volatility: 19.73% vs 19.02% current
- ✅ Walk-forward validation: MAE 0.068, RMSE 0.088 across 5 folds
- ✅ SHAP top drivers: ret_21d, macro_CPIAUCSL, macro_CPIAUCSL_roc

**Sprint 4 — RAG Engine**
- ✅ 1,802 chunks embedded across 10 companies
- ✅ GS risk factors query: similarity 0.616 from 2025 10-K
- ✅ BLK revenue model query: similarity 0.624 from 2024 10-K
- ✅ Source URLs traceable to actual SEC EDGAR filings

**Sprint 5 — LLM Reasoning Layer**
- ✅ /chat returning grounded institutional-grade prose analysis
- ✅ /chat/stress-test: Fed 200bps scenario producing four-paragraph risk report
- ✅ Sources cited: JPM 2026 10-K, GS 2026 10-K, BLK 2024 10-K
- ✅ Full pipeline verified: quant → ML → RAG → LLM synthesis
---

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- Free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
- Free Gemini API key: https://aistudio.google.com/app/apikey

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/portfolioiq.git
cd portfolioiq
cp .env.example .env
# Edit .env — add your FRED_API_KEY and GEMINI_API_KEY
docker-compose up --build
```

### Load data

```bash
# In a second terminal
docker-compose exec api python scripts/run_ingestion.py
docker-compose exec api python scripts/run_doc_ingestion.py
```

### Try it

Open `http://localhost:8000/docs` for interactive Swagger UI, or:

```bash
# Full risk report
curl -X POST http://localhost:8000/risk/analyse \
  -H "Content-Type: application/json" \
  -d '{"weights":{"AAPL":0.3,"MSFT":0.2,"GS":0.2,"BLK":0.15,"JNJ":0.15}}'

# The flagship endpoint
curl -X POST http://localhost:8000/chat/stress-test \
  -H "Content-Type: application/json" \
  -d '{"scenario":"Fed raises rates 200bps","weights":{"AAPL":0.3,"MSFT":0.2,"GS":0.2,"BLK":0.15,"JNJ":0.15}}'
```

---

## Project Structure

```
portfolioiq/
├── app/
│   ├── core/config.py           # Settings and environment variables
│   ├── db/database.py           # SQLAlchemy engine and session management
│   ├── models/
│   │   ├── market_data.py       # Tickers, prices, macro indicators
│   │   ├── portfolio.py         # Portfolios and holdings
│   │   └── documents.py         # Document chunks with vector embeddings
│   ├── ingestion/
│   │   ├── prices.py            # Yahoo Finance ingestion
│   │   ├── macro.py             # FRED macroeconomic ingestion
│   │   └── documents.py         # SEC EDGAR filing ingestion
│   ├── services/
│   │   ├── quant_engine.py      # Quantitative analytics engine
│   │   ├── ml_engine.py         # XGBoost forecasting + SHAP
│   │   ├── rag_engine.py        # Document retrieval engine
│   │   └── llm_engine.py        # LLM reasoning and synthesis
│   ├── api/routes.py            # FastAPI endpoint handlers
│   └── main.py                  # Application entrypoint
├── scripts/
│   ├── run_ingestion.py         # Trigger price + macro data pull
│   └── run_doc_ingestion.py     # Trigger document embedding pipeline
├── tests/test_api.py
├── docker-compose.yml
├── Dockerfile
├── init.sql                     # Enables pgvector on first boot
└── requirements.txt
```

## Roadmap

- **Sprint 6** — React dashboard: portfolio input, risk visualisation, stress test interface, chat UI
- Portfolio optimisation: Modern Portfolio Theory rebalancing suggestions
- Real-time news integration as a live ML feature
- Multi-asset support: fixed income and commodities
- AWS deployment: EC2, nginx, HTTPS, GitHub Actions CI/CD

---

## A Note on Yahoo Finance Rate Limiting

Yahoo Finance applies aggressive rate limiting to automated requests. The ingestion pipeline handles this with built-in retry logic, exponential backoff, and inter-ticker delays. If you see 429 errors, wait 15-20 minutes before retrying — this is a Yahoo-side constraint, not a bug in this codebase.