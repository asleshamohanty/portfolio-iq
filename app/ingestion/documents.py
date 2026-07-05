"""
app/ingestion/documents.py

Fetches 10-K and 10-Q filings from SEC EDGAR for a list of companies,
chunks them into 500-token segments with 50-token overlap, embeds
each chunk using sentence-transformers, and stores in Postgres.

Run: docker-compose exec api python scripts/run_doc_ingestion.py
"""

import time
import re
import requests
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.database import SessionLocal
from app.models.documents import DocumentChunk
from app.core.config import settings

# 15 companies — narrow and deep, quality over volume
TARGET_COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "JPM":  "0000019617",
    "GS":   "0000886982",
    "BLK":  "0001364742",
    "V":    "0001403161",
    "JNJ":  "0000200406",
}

EDGAR_BASE = "https://data.sec.gov/submissions"
HEADERS = {"User-Agent": "PortfolioIQ research@portfolioiq.dev"}

CHUNK_SIZE    = 500   # tokens (approximated as words here)
CHUNK_OVERLAP = 50


def _get_embedder():
    """Load sentence-transformers model (downloads once, cached after)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    500 words ≈ 500 tokens for English financial text.
    Overlap ensures context isn't lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 100:  # skip tiny fragments
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _clean_text(text: str) -> str:
    """Basic cleanup — remove excessive whitespace and boilerplate markers."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()


def _fetch_recent_filings(cik: str, form_type: str, limit: int = 2) -> list[dict]:
    """
    Hit SEC EDGAR's submissions API to get recent filing metadata.
    Returns list of {accession_number, filing_date, primary_document}.
    """
    url = f"{EDGAR_BASE}/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    ! EDGAR submissions fetch failed for CIK {cik}: {e}")
        return []

    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates   = filings.get("filingDate", [])
    docs    = filings.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form == form_type and len(results) < limit:
            results.append({
                "accession": accessions[i].replace("-", ""),
                "date": dates[i],
                "doc": docs[i],
                "accession_raw": accessions[i],
            })
    return results


def _fetch_filing_text(cik: str, accession: str, doc: str) -> str:
    """Download the actual filing document text from EDGAR."""
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        # Strip HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        return _clean_text(text)
    except Exception as e:
        print(f"    ! Failed to fetch filing text: {e}")
        return ""


def ingest_company(symbol: str, cik: str, embedder, form_type: str = "10-K"):
    """Full pipeline for one company: fetch → chunk → embed → store."""
    print(f"  Processing {symbol} ({form_type})...")
    filings = _fetch_recent_filings(cik, form_type, limit=2)

    if not filings:
        print(f"    ! No {form_type} filings found for {symbol}")
        return 0

    db = SessionLocal()
    total_chunks = 0

    for filing in filings:
        print(f"    Filing: {filing['date']}...")
        text = _fetch_filing_text(cik, filing["accession"], filing["doc"])

        if not text or len(text) < 500:
            print(f"    ! Text too short or empty, skipping")
            continue

        # Limit to first 50k words to keep processing time reasonable
        words = text.split()[:50000]
        text = " ".join(words)

        chunks = _chunk_text(text)
        print(f"    Embedding {len(chunks)} chunks...")

        embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False)

        for chunk_text, embedding in zip(chunks, embeddings):
            stmt = pg_insert(DocumentChunk).values(
                symbol=symbol,
                doc_type=form_type,
                period=filing["date"][:7],  # "2023-11"
                source_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{filing['accession']}/{filing['doc']}",
                chunk_text=chunk_text,
                embedding=embedding.tolist(),
            )
            # If same chunk already exists for this symbol+period, skip
            stmt = stmt.on_conflict_do_nothing()
            db.execute(stmt)

        db.commit()
        total_chunks += len(chunks)
        time.sleep(1)  # be polite to EDGAR

    db.close()
    print(f"    -> {total_chunks} chunks stored for {symbol}")
    return total_chunks


def run_all():
    """Ingest 10-K filings for all target companies."""
    print("Loading embedder (downloads ~90MB on first run)...")
    embedder = _get_embedder()
    print("Embedder ready.")

    total = 0
    for symbol, cik in TARGET_COMPANIES.items():
        total += ingest_company(symbol, cik, embedder, form_type="10-K")
        time.sleep(2)  # EDGAR rate limit: polite pacing between companies

    print(f"\nDone. {total} total chunks stored across {len(TARGET_COMPANIES)} companies.")


if __name__ == "__main__":
    run_all()