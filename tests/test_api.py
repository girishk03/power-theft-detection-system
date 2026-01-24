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
