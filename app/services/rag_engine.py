"""
app/services/rag_engine.py

Sprint 4 — RAG Engine.

Given a user question:
  1. Embed the question with the same model used at ingestion
  2. Find the top-K most similar chunks via pgvector cosine search
  3. Return chunks with source metadata for the LLM layer (Sprint 5)

Kept deliberately simple — quality retrieval, no hallucination.
"""

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.documents import DocumentChunk


SIMILARITY_THRESHOLD = 0.30   # reject chunks below this cosine similarity
TOP_K = 5


def _get_embedder():
    """Same model as ingestion — must match or similarity is meaningless."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def search_documents(
    db: Session,
    query: str,
    symbol: str = None,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Semantic search over document_chunks using pgvector cosine similarity.

    Args:
        query   : plain-English question e.g. "What are NVIDIA's main risk factors?"
        symbol  : optional — restrict search to one company
        top_k   : number of chunks to return
        threshold: minimum cosine similarity (0-1). Below this = not relevant enough.

    Returns:
        List of dicts with chunk_text, symbol, doc_type, period, similarity score.
    """
    embedder = _get_embedder()
    query_embedding = embedder.encode([query])[0].tolist()

    # pgvector's <=> operator = cosine distance (0=identical, 2=opposite)
    # cosine similarity = 1 - cosine distance
    if symbol:
        sql = text("""
            SELECT
                chunk_text,
                symbol,
                doc_type,
                period,
                source_url,
                1 - (embedding <=> cast(:embedding as vector)) AS similarity
            FROM document_chunks
            WHERE symbol = :symbol
            ORDER BY embedding <=> cast(:embedding as vector)
            LIMIT :top_k
        """)
        rows = db.execute(sql, {
            "embedding": str(query_embedding),
            "symbol": symbol.upper(),
            "top_k": top_k,
        }).fetchall()
    else:
        sql = text("""
            SELECT
                chunk_text,
                symbol,
                doc_type,
                period,
                source_url,
                1 - (embedding <=> cast(:embedding as vector)) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> cast(:embedding as vector)
            LIMIT :top_k
        """)
        rows = db.execute(sql, {
            "embedding": str(query_embedding),
            "top_k": top_k,
        }).fetchall()

    results = []
    for row in rows:
        similarity = float(row.similarity)
        if similarity < threshold:
            continue  # not relevant enough — don't mislead the LLM
        results.append({
            "chunk_text": row.chunk_text,
            "symbol":     row.symbol,
            "doc_type":   row.doc_type,
            "period":     row.period,
            "source_url": row.source_url,
            "similarity": round(similarity, 4),
        })

    return results