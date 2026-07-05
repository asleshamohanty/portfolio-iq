"""
app/models/documents.py

One table: document_chunks
Each row = one chunk of a financial document + its embedding vector.

We use pgvector's VECTOR type for the embedding column so Postgres
can do cosine similarity search directly — no separate vector DB needed.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.db.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id         = Column(Integer, primary_key=True)
    symbol     = Column(String(10), index=True)        # e.g. "AAPL"
    doc_type   = Column(String(20))                    # "10-K", "10-Q", "earnings"
    period     = Column(String(20))                    # e.g. "2023-Q4"
    source_url = Column(String(500))
    chunk_text = Column(Text, nullable=False)
    embedding  = Column(Vector(384))                   # 384-dim: all-MiniLM-L6-v2
    created_at = Column(DateTime(timezone=True), server_default=func.now())