import os
import sys

import pytest


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    import app as app_module

    app_module.app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
    })
    app_module.DATA_MODE = "simulated"
    app_module.df_extended = None

    with app_module.app.test_client() as test_client:
        with test_client.session_transaction() as session:
            session["logged_in"] = True
            session["username"] = "admin"
        yield test_client

    app_module.df_extended = None
