"""
app/db/database.py

This sets up the SQLAlchemy "engine" (the thing that actually talks
to Postgres) and a session factory. FastAPI routes will call
get_db() to borrow a session for the duration of one request.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# All model classes (Ticker, DailyPrice, etc.) inherit from this
Base = declarative_base()


def get_db():
    """
    FastAPI dependency. Yields a DB session and guarantees it's
    closed afterward, even if the request raises an error.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
