"""
app/main.py

The entrypoint. Run with:
  uvicorn app.main:app --reload

This:
  1. Creates all tables on startup if they don't exist yet
     (fine for dev; in Sprint 6 you'd use Alembic migrations instead)
  2. Wires up the routes from app/api/routes.py
  3. Auto-generates docs at /docs (Swagger UI) and /redoc
"""

from fastapi import FastAPI

from app.db.database import engine, Base
from app.api.routes import router
from app.models import market_data  # noqa: F401 — import so tables register with Base

app = FastAPI(
    title="PortfolioIQ",
    description="AI-powered portfolio intelligence platform — Sprint 1: Data Foundation",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "PortfolioIQ API — Sprint 1 (Data Foundation)",
        "docs": "/docs",
    }
