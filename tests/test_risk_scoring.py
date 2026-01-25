from src.risk_scoring import compute_risk_score


def test_compute_risk_score_bounds_default_range():
    score = compute_risk_score(10, 20)
    assert 0 <= score <= 1


def test_compute_risk_score_clamps_low():
    score = compute_risk_score(1000, 10)
    assert score == 0.3


def test_compute_risk_score_clamps_high():
    score = compute_risk_score(0, 10)
    assert score == 0.95


def test_compute_risk_score_default_on_invalid_expected():
    score = compute_risk_score(10, 0)
    assert score == 0.8


def test_compute_risk_score_default_on_bad_inputs():
    score = compute_risk_score("x", "y")
    assert score == 0.8
