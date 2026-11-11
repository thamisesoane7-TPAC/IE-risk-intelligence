import importlib
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ieri_risk.api import demo, router


@pytest.fixture
def client(trained, monkeypatch):
    monkeypatch.setenv("IERI_MODEL_DIR", str(trained["registry"]))
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


def test_score_explain_metadata_and_history(client):
    payload = demo(0)
    score = client.post("/risk/score", json=payload["score_request"])
    assert score.status_code == 200, score.text
    result = score.json()
    history = client.post("/risk/score-history", json=payload["history_request"])
    assert history.status_code == 200, history.text
    assert history.json()["synthetic_stress_probability_30d"] == result["synthetic_stress_probability_30d"]
    explained = client.post("/risk/explain", json=payload["score_request"])
    assert explained.status_code == 200, explained.text
    assert abs(explained.json()["explanation"]["reconstructed_probability"] - result["synthetic_stress_probability_30d"]) < 1e-6
    assert client.get("/risk/metadata").json()["model_id"] == result["model_id"]
    assert client.get("/risk/evaluation").status_code == 200
    assert client.get("/risk/health").status_code == 200
    assert client.get("/risk/schema").status_code == 200


@pytest.mark.parametrize("mutation", ["extra", "missing", "string", "schema", "boolean", "inconsistent"])
def test_invalid_feature_requests_422(client, mutation):
    payload = demo(0)["score_request"]
    if mutation == "extra":
        payload["features"]["age"] = 24
    elif mutation == "missing":
        payload["features"].pop("liquidity_coverage_days")
    elif mutation == "string":
        payload["features"]["digital_payment_share"] = "0.5"
    elif mutation == "schema":
        payload["schema_version"] = "legacy"
    elif mutation == "boolean":
        payload["features"]["digital_payment_share"] = True
    else:
        payload["features"]["net_cash_flow_zar"] += 100
    assert client.post("/risk/score", json=payload).status_code == 422


def test_nonfinite_json_rejected_without_server_crash(client):
    import json
    payload = demo(0)["score_request"]
    payload["features"]["digital_payment_share"] = float("nan")
    response = client.post("/risk/score", content=json.dumps(payload), headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_history_date_and_business_validation(client):
    payload = demo(0)["history_request"]
    payload["history"][1]["business_id"] = "wrong"
    assert client.post("/risk/score-history", json=payload).status_code == 422


def test_untrained_api_is_explicit_503(tmp_path, monkeypatch):
    monkeypatch.setenv("IERI_MODEL_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/risk/health").status_code == 503
    assert client.post("/risk/score", json=demo(0)["score_request"]).status_code == 503
    assert client.get("/risk/schema").status_code == 200


def test_wrapper_preserves_original_app_and_routes(monkeypatch):
    base = types.ModuleType("ieri_test_base_app")
    base.app = FastAPI()
    @base.app.get("/existing-businesses")
    def existing():
        return [{"id": 1}]
    monkeypatch.setitem(sys.modules, "ieri_test_base_app", base)
    monkeypatch.setenv("IERI_BASE_APP", "ieri_test_base_app:app")
    monkeypatch.delitem(sys.modules, "ieri_asgi", raising=False)
    wrapper = importlib.import_module("ieri_asgi")
    assert wrapper.app is base.app
    client = TestClient(wrapper.app)
    assert client.get("/existing-businesses").json() == [{"id": 1}]
    assert client.get("/risk/schema").status_code == 200
    number_of_routes = len(wrapper.app.routes)
    importlib.reload(wrapper)
    assert len(wrapper.app.routes) == number_of_routes
    assert "/risk/schema" in wrapper.app.openapi()["paths"]


def test_wrapper_rejects_conflicting_routes(monkeypatch):
    base = types.ModuleType("ieri_conflicting_app")
    base.app = FastAPI()
    @base.app.get("/risk/score")
    def existing():
        return {}
    monkeypatch.setitem(sys.modules, "ieri_conflicting_app", base)
    monkeypatch.setenv("IERI_BASE_APP", "ieri_conflicting_app:app")
    monkeypatch.delitem(sys.modules, "ieri_asgi", raising=False)
    with pytest.raises(RuntimeError, match="conflicts"):
        importlib.import_module("ieri_asgi")
