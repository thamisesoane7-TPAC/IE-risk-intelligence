import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def build_engine(database_url: str | None = None) -> Engine:
    url = database_url or os.getenv(
        "DATABASE_URL", "sqlite:///./data/ieri.db"
    )

    options = {"pool_pre_ping": True}

    if url.startswith("sqlite"):
        options["connect_args"] = {
            "check_same_thread": False
        }

        if url in {"sqlite://", "sqlite:///:memory:"}:
            options["poolclass"] = StaticPool

    return create_engine(url, **options)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False
    )
