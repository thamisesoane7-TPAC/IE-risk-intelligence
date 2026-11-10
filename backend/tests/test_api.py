from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    with TestClient(create_app("sqlite://")) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_business_create_and_read():
    with TestClient(create_app("sqlite://")) as client:
        payload = {
            "name": "Demo Spaza",
            "sector": "spaza_shop",
            "province": "Gauteng"
        }

        created = client.post(
            "/api/v1/businesses",
            json=payload
        )

        assert created.status_code == 201

        business_id = created.json()["id"]

        response = client.get(
            f"/api/v1/businesses/{business_id}"
        )

        assert response.json()["name"] == "Demo Spaza"

        businesses = client.get(
            "/api/v1/businesses"
        ).json()

        assert len(businesses) == 1


def test_invalid_sector_is_rejected():
    with TestClient(create_app("sqlite://")) as client:
        response = client.post(
            "/api/v1/businesses",
            json={
                "name": "Demo Spaza",
                "sector": "unknown",
                "province": "Gauteng"
            }
        )

        assert response.status_code == 422


def test_unknown_business_is_404():
    with TestClient(create_app("sqlite://")) as client:
        response = client.get("/api/v1/businesses/999")
        assert response.status_code == 404
