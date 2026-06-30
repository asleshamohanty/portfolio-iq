"""
app/core/config.py

Central configuration. Everything that varies between your laptop,
a teammate's laptop, and a production server lives here — pulled
from environment variables, never hardcoded.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Postgres connection — these match docker-compose.yml service values
    POSTGRES_USER: str = "portfolioiq"
    POSTGRES_PASSWORD: str = "portfolioiq_dev_password"
    POSTGRES_DB: str = "portfolioiq"
    POSTGRES_HOST: str = "localhost"   # becomes "db" when running inside Docker
    POSTGRES_PORT: int = 5432

    # FRED API (free key from https://fred.stlouisfed.org/docs/api/api_key.html)
    FRED_API_KEY: str = ""

    # Which tickers to track in Sprint 1 (keep this small to start)
    DEFAULT_TICKERS: str = "AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,GS,BLK,V,JNJ"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def ticker_list(self) -> list[str]:
        return [t.strip().upper() for t in self.DEFAULT_TICKERS.split(",")]

    class Config:
        env_file = ".env"


settings = Settings()
