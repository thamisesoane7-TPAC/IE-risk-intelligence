from contextlib import asynccontextmanager
from collections.abc import Generator
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import (
    Base,
    build_engine,
    get_session_factory
)
from app.models import Business
from app.services.synthetic import (
    PROFILES,
    generate_ledger,
)
from app.services.features import calculate_features


Sector = Literal[
    "spaza_shop",
    "food_vendor",
    "salon",
    "repair_shop",
    "other"
]


class BusinessCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    sector: Sector
    province: str = Field(min_length=2, max_length=60)


class BusinessRead(BusinessCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int


def create_app(database_url: str | None = None) -> FastAPI:
    engine = build_engine(database_url)
    SessionLocal = get_session_factory(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        Base.metadata.create_all(bind=engine)
        yield
        engine.dispose()

    app = FastAPI(
        title="IERI — Informal Economy Risk Intelligence",
        version="0.1.0",
        lifespan=lifespan
    )

    def get_db() -> Generator[Session, None, None]:
        with SessionLocal() as db:
            yield db

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "ieri-api",
            "version": "0.1.0"
        }

    @app.post(
        "/api/v1/businesses",
        response_model=BusinessRead,
        status_code=status.HTTP_201_CREATED
    )
    def create_business(
        payload: BusinessCreate,
        db: Session = Depends(get_db)
    ) -> Business:
        business = Business(**payload.model_dump())
        db.add(business)
        db.commit()
        db.refresh(business)
        return business

    @app.get(
        "/api/v1/businesses",
        response_model=list[BusinessRead]
    )
    def list_businesses(
        db: Session = Depends(get_db)
    ) -> list[Business]:
        return list(
            db.scalars(
                select(Business)
                .order_by(Business.id)
                .limit(100)
            )
        )

    @app.get(
        "/api/v1/businesses/{business_id}",
        response_model=BusinessRead
    )
    def read_business(
        business_id: int,
        db: Session = Depends(get_db)
    ) -> Business:
        business = db.get(Business, business_id)

        if business is None:
            raise HTTPException(
                status_code=404,
                detail="Business not found"
            )

        return business


    @app.get("/api/v1/demo-businesses")
    def list_demo_businesses() -> list[dict[str, str]]:
        return [
            {
                "business_id": str(p["id"]),
                "sector": str(p["sector"]),
            }
            for p in PROFILES
        ]

    @app.get(
        "/api/v1/demo-businesses/{business_id}/features"
    )
    def demo_business_features(
        business_id: str,
    ) -> dict:
        known_ids = {p["id"] for p in PROFILES}

        if business_id not in known_ids:
            raise HTTPException(
                status_code=404,
                detail="Demo business not found",
            )

        ledger = generate_ledger(seed=42, days=120)
        return calculate_features(
            ledger,
            business_id,
        )

    return app


app = create_app()
