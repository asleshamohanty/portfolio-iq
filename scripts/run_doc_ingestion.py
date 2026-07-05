"""
scripts/run_doc_ingestion.py

Run SEC filing ingestion for all target companies.
Takes 5-10 minutes on first run (embedding is CPU-bound).

Usage:
  docker-compose exec api python scripts/run_doc_ingestion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.documents import run_all

if __name__ == "__main__":
    print("=" * 50)
    print("PortfolioIQ — Document Ingestion (Sprint 4)")
    print("=" * 50)
    run_all()