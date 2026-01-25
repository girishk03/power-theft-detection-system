import os

import pytest


@pytest.fixture()
def client():
    os.environ.setdefault("API_KEY", "test-key")
    from app import app  # noqa: WPS433

    app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
    })

    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "admin"
        yield client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json["status"] == "ok"


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
        assert 0 <= row["risk_score"] <= 1


def test_invalid_year_returns_400(client):
    res = client.get("/api/year-detections/2010", headers={"X-API-KEY": "test-key"})
    assert res.status_code == 400
    assert res.json["error"] == "Invalid year"


def test_api_requires_login_session():
    os.environ.setdefault("API_KEY", "test-key")
    from app import app  # noqa: WPS433

    app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
    })

    with app.test_client() as client:
        res = client.get("/api/available-years", headers={"X-API-KEY": "test-key"})
        assert res.status_code in {301, 302}
        assert "/login" in res.headers.get("Location", "")
