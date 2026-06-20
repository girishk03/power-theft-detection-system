from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app as app_module
from src.data_quality import summarize_dataset
from src.risk_scoring import compute_risk_score


RAW_DATASET = Path(__file__).resolve().parents[1] / "data" / "raw" / "Electricity_Theft_Data.csv"


@pytest.fixture(autouse=True)
def reset_dataset_cache():
    original_mode = app_module.DATA_MODE
    original_path = app_module.DATASET_PATH
    app_module.df_extended = None
    yield
    app_module.DATA_MODE = original_mode
    app_module.DATASET_PATH = original_path
    app_module.df_extended = None


def test_real_data_mode_loads_first_1000_rows():
    app_module.DATA_MODE = "real"
    app_module.DATASET_PATH = str(RAW_DATASET)

    loaded = app_module.load_extended_dataset()
    expected = pd.read_csv(RAW_DATASET, nrows=1000)

    assert loaded is not None
    assert len(loaded) == 1000
    pd.testing.assert_frame_equal(loaded, expected)


def test_included_dataset_quality_counts_match_documentation():
    dataframe = pd.read_csv(RAW_DATASET)

    summary = summarize_dataset(dataframe)

    assert summary == {
        "customer_rows": 9957,
        "daily_columns": 365,
        "possible_readings": 3634305,
        "present_readings": 3140639,
        "missing_readings": 493666,
        "missing_rate": pytest.approx(0.1358350496),
        "normal_labels": 8562,
        "theft_labels": 1394,
        "missing_labels": 1,
    }


def test_real_data_statistics_handle_missing_values_safely(tmp_path):
    dataset = pd.DataFrame({
        "CONS_NO": range(4),
        "01/01/2015": [10.0, np.nan, 30.0, 40.0],
        "02/01/2015": [12.0, 20.0, np.nan, 42.0],
        "CHK_STATE": [0, 1, 0, np.nan],
    })
    dataset_path = tmp_path / "readings.csv"
    dataset.to_csv(dataset_path, index=False)
    app_module.DATA_MODE = "real"
    app_module.DATASET_PATH = str(dataset_path)

    with app_module.app.test_request_context("/api/year-statistics/2015"):
        response = app_module.get_year_statistics.__wrapped__.__wrapped__(2015)
        payload = response.get_json()

    assert payload["year"] == 2015
    assert payload["avg_consumption"] == pytest.approx(25.6667, rel=1e-3)
    assert payload["data_available"] is True


def test_risk_scoring_accepts_real_dataset_sample_rows():
    dataframe = pd.read_csv(RAW_DATASET, nrows=1000)
    reading_columns = [column for column in dataframe.columns if column not in {"CONS_NO", "CHK_STATE"}]
    customer_means = dataframe[reading_columns].mean(axis=1, skipna=True)
    actual = float(customer_means.dropna().iloc[0])
    expected = float(customer_means.dropna().median())

    score = compute_risk_score(actual, expected)
    expected_score = min(0.95, max(0.30, 1 - (actual / expected)))

    assert score == pytest.approx(expected_score)
    assert 0.30 <= score <= 0.95
