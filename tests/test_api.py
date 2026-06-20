import pytest


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json["status"] == "ok"


def test_dashboard_route_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Power Theft" in res.data


def test_api_requires_key_when_configured(client):
    res = client.get("/api/available-years")
    assert res.status_code == 401

    res = client.get("/api/available-years", headers={"X-API-KEY": "test-key"})
    assert res.status_code == 200
    assert "years" in res.json


def test_risk_score_bounds(client):
    res = client.get("/api/year-detections/2024", headers={"X-API-KEY": "test-key"})
    assert res.status_code == 200
    assert isinstance(res.json, list)

    for row in res.json:
        assert "risk_score" in row
        assert "risk_level" in row
        assert "meter_id" in row
        assert 0 <= row["risk_score"] <= 1


@pytest.mark.parametrize(
    ("route", "expected_key"),
    [
        ("/api/available-years", "years"),
        ("/api/year-statistics/2024", "total_customers"),
        ("/api/year-consumption/2024", "consumption"),
        ("/api/year-consumption/2024/MTR-2024-00001", "consumption"),
    ],
)
def test_documented_api_routes(client, route, expected_key):
    res = client.get(route, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 200
    assert expected_key in res.json


def test_non_numeric_year_returns_404(client):
    res = client.get("/api/year-detections/not-a-year", headers={"X-API-KEY": "test-key"})
    assert res.status_code == 404


def test_invalid_year_returns_400(client):
    res = client.get("/api/year-detections/2010", headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.json["error"] == "Invalid year"


@pytest.mark.parametrize(
    "route",
    [
        "/api/year-statistics/2010",
        "/api/year-consumption/2026",
        "/api/year-detections/2010",
    ],
)
def test_year_endpoints_reject_out_of_range_values(client, route):
    res = client.get(route, headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.json["error"] == "Invalid year"


def test_api_requires_login_session():
    import os

    os.environ.setdefault("API_KEY", "test-key")
    from app import app

    app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
    })

    with app.test_client() as client:
        res = client.get("/api/available-years", headers={"X-API-KEY": "test-key"})
        assert res.status_code in {301, 302}
        assert "/login" in res.headers.get("Location", "")
