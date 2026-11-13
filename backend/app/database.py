"""Database engine, session management, and table initialization."""

from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import get_settings

settings = get_settings()

# Ensure parent directory for SQLite file exists
if settings.DATABASE_URL.startswith("sqlite:///"):
    sqlite_path = settings.DATABASE_URL.replace("sqlite:///", "")
    db_file_path = Path(sqlite_path)
    if not db_file_path.is_absolute():
        db_file_path = Path.cwd() / db_file_path
    db_file_path.parent.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides an active SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    # Import db_models to register them on Base.metadata
    from app.models import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
