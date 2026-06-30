"""
tests/test_api.py

Basic smoke tests for Sprint 1. Run with: pytest

Note: these expect the API to be reachable and the DB to have been
seeded via scripts/run_ingestion.py first — they're integration
tests, not pure unit tests.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_tickers_endpoint_returns_list():
    response = client.get("/tickers")
    assert response.status_code == 200
    body = response.json()
    assert "data" in body
    assert "count" in body
    assert isinstance(body["data"], list)


def test_prices_for_unknown_symbol_returns_404():
    response = client.get("/prices/ZZZZZZ")
    assert response.status_code == 404


def test_macro_for_unknown_series_returns_404():
    response = client.get("/macro/NOT_A_REAL_SERIES")
    assert response.status_code == 404
