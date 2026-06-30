"""
scripts/run_ingestion.py

Manually triggers a full data pull: prices for every watchlist
ticker, plus all macro series. Run this after docker-compose is up:

  python scripts/run_ingestion.py
"""

import sys
import os

# Allow running this script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import prices, macro


def main():
    print("=" * 50)
    print("PortfolioIQ — Data Ingestion")
    print("=" * 50)

    print("\n[1/2] Ingesting price data...")
    prices.run_all()

    print("\n[2/2] Ingesting macro data...")
    macro.run_all()

    print("\nIngestion complete. Try: curl http://localhost:8000/tickers")


if __name__ == "__main__":
    main()
