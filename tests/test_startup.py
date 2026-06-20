def test_flask_app_imports_with_documented_routes():
    from app import app

    routes = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/" in routes
    assert "/health" in routes
    assert "/api/year-detections/<int:year>" in routes


def test_app_starts_in_test_mode():
    from app import app

    app.config.update(TESTING=True, SECRET_KEY="startup-test")
    with app.test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_environment_defaults_match_documentation():
    import app as app_module

    assert app_module.DATA_MODE == "simulated"
    assert app_module.DATASET_PATH == "data/processed/sgcc_extended_2014_2025.csv"
