import math


def compute_risk_score(actual_consumption, expected_consumption, *, min_score=0.3, max_score=0.95, default_score=0.8):
    if expected_consumption is None:
        return float(default_score)

    try:
        expected = float(expected_consumption)
        actual = float(actual_consumption)
    except (TypeError, ValueError):
        return float(default_score)

    if not math.isfinite(expected) or expected <= 0:
        return float(default_score)

    if not math.isfinite(actual):
        return float(default_score)

    score = 1 - (actual / expected)
    if score < min_score:
        return float(min_score)
    if score > max_score:
        return float(max_score)
    return float(score)
