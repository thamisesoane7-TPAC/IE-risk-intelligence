"""Namespaced, read-only model endpoints; training is an offline CLI operation."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from .contracts import FinancialFeatures, HistoryRequest, NOTICE, SCHEMA_VERSION, ScoreRequest
from .features import calculate_features, validate_history
from .service import RiskService
from .simulation import simulate_business

logger = logging.getLogger("ieri.risk")


class RiskRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def safe_validation(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as error:
                # Never echo raw financial input or non-JSON NaN/Infinity in validation errors.
                return JSONResponse(status_code=422, content={"detail": [
                    {"type": e["type"], "loc": e["loc"], "msg": e["msg"]} for e in error.errors()]})
        return safe_validation


router = APIRouter(prefix="/risk", tags=["IERI quantitative risk"], route_class=RiskRoute)


@lru_cache(maxsize=2)
def _cached_service(path: str, pointer_text: str):
    return RiskService(Path(path))


def get_service() -> RiskService:
    path = Path(os.environ.get("IERI_MODEL_DIR", Path(__file__).resolve().parents[1] / "artifacts" / "ieri")).resolve()
    try:
        pointer = (path / "CURRENT.json").read_text()
        json.loads(pointer)
        return _cached_service(str(path), pointer)
    except Exception:
        logger.exception("IERI model registry could not be loaded")
        raise HTTPException(status_code=503, detail={
            "code": "MODEL_UNAVAILABLE", "message": "Train or verify the trusted local model registry; inspect server logs."}) from None


def guarded_score(request: ScoreRequest, *, explanations: bool = False):
    service = get_service()
    try:
        return service.explanation(request) if explanations else service.score(request)
    except Exception:
        logger.exception("IERI scoring or explanation failed")
        raise HTTPException(status_code=503, detail={"code": "SCORING_UNAVAILABLE", "message": "The model could not produce a verified result."}) from None


@router.get("/health")
def health():
    service = get_service()
    return {"status": "ready", "model_id": service.metadata["model_id"], "notice": NOTICE}


@router.get("/schema")
def schema():
    return {"schema_version": SCHEMA_VERSION, "history_days": 90,
            "features": FinancialFeatures.model_json_schema(), "request": ScoreRequest.model_json_schema(), "notice": NOTICE}


@router.get("/metadata")
def metadata():
    return get_service().metadata


@router.get("/evaluation")
def evaluation():
    return get_service().report


@router.post("/score")
def score(request: ScoreRequest):
    """Score one explicitly versioned financial snapshot; business_id is a caller reference, not a DB lookup."""
    return guarded_score(request)


@router.post("/explain")
def explain_score(request: ScoreRequest):
    """Explain the exact served calibrated score with deterministic permutation SHAP."""
    return guarded_score(request, explanations=True)


@router.post("/score-history")
def score_history(request: HistoryRequest):
    """Reconcile 90 daily records and compute the same features used during training."""
    try:
        history = validate_history(pd.DataFrame([r.model_dump() for r in request.history]))
        payload = ScoreRequest(schema_version=SCHEMA_VERSION, business_id=history.business_id.iloc[0],
                               observation_end=history.date.iloc[-1].date(), features=calculate_features(history))
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="History must contain one business, 90 consecutive days and reconciled cash/arrears balances") from None
    return {**guarded_score(payload), "features": payload.features.model_dump(), "feature_source": "validated_daily_history"}


@router.get("/demo")
def demo(index: int = Query(default=0, ge=0, le=99)):
    """Fictional input example; this endpoint does not persist or register a real business."""
    ledger, _ = simulate_business(index, seed=81231)
    history = ledger.iloc[:90].copy()
    payload = ScoreRequest(schema_version=SCHEMA_VERSION, business_id=history.business_id.iloc[0],
                           observation_end=history.date.iloc[-1], features=calculate_features(history))
    return {"score_request": payload.model_dump(mode="json"), "history_request": {"history": history.to_dict("records")}, "notice": NOTICE}
