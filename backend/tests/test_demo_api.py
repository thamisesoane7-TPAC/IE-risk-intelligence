from fastapi.testclient import TestClient

from app.main import create_app


def test_demo_business_list():
    with TestClient(create_app("sqlite://")) as client:
        response = client.get(
            "/api/v1/demo-businesses"
        )

        assert response.status_code == 200
        assert len(response.json()) == 5


def test_demo_financial_features():
    with TestClient(create_app("sqlite://")) as client:
        response = client.get(
            "/api/v1/demo-businesses/IERI-001/features"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["business_id"] == "IERI-001"
        assert data["observation_days"] == 120
        assert data["average_daily_sales_zar"] > 0
        assert 0 <= data["digital_payment_share"] <= 1


def test_unknown_demo_business():
    with TestClient(create_app("sqlite://")) as client:
        response = client.get(
            "/api/v1/demo-businesses/UNKNOWN/features"
        )

        assert response.status_code == 404
