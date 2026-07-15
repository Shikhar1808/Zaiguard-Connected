"""
ZaiGuard Alert Engine — Application Settings
=============================================
Single source of truth for all configuration values.
Loaded from environment variables / .env file via pydantic-settings.

Usage anywhere in the codebase:
    from config.settings import settings
    db_url = settings.database_url
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All values are read from environment variables.
    Fallbacks are development-safe defaults only — never use
    these defaults in production.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─────────────────────────────────────────────────────────
    # PostgreSQL / TimescaleDB
    # ─────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "zaiguard"
    postgres_password: str = "zaiguard_dev"
    postgres_db: str = "zaiguard"

    @property
    def database_url(self) -> str:
        """
        Async SQLAlchemy connection string.
        Uses asyncpg driver for non-blocking database I/O.
        FastAPI is async — blocking DB calls would defeat the purpose.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ─────────────────────────────────────────────────────────
    # Redis
    # ─────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ─────────────────────────────────────────────────────────
    # Qdrant
    # ─────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_dismissed: str = "dismissed_alerts"
    qdrant_collection_confirmed: str = "confirmed_alerts"

    # ─────────────────────────────────────────────────────────
    # Embedding model
    # ─────────────────────────────────────────────────────────
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_cache_dir: str = "./.model_cache"

    # ─────────────────────────────────────────────────────────
    # Config cache
    # ─────────────────────────────────────────────────────────
    config_cache_ttl_seconds: int = 60

    # ─────────────────────────────────────────────────────────
    # Application
    # ─────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "DEBUG"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    lru_cache ensures this is a singleton — the .env file is read
    once at startup, not on every function call.

    FastAPI dependency injection usage:
        from config.settings import get_settings
        from fastapi import Depends

        @router.get("/health")
        def health(settings: Settings = Depends(get_settings)):
            ...
    """
    return Settings()


# Module-level singleton for non-DI usage
settings = get_settings()