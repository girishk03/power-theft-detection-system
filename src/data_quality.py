from __future__ import annotations

import pandas as pd


def summarize_dataset(
    dataframe: pd.DataFrame,
    *,
    customer_column: str = "CONS_NO",
    label_column: str = "CHK_STATE",
) -> dict[str, int | float]:
    """Return deterministic completeness and label counts for a customer-reading table."""
    required_columns = {customer_column, label_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {names}")

    reading_columns = [
        column
        for column in dataframe.columns
        if column not in {customer_column, label_column}
    ]
    readings = dataframe[reading_columns]
    possible_readings = len(dataframe) * len(reading_columns)
    present_readings = int(readings.notna().sum().sum())
    missing_readings = int(readings.isna().sum().sum())
    labels = dataframe[label_column]

    return {
        "customer_rows": len(dataframe),
        "daily_columns": len(reading_columns),
        "possible_readings": possible_readings,
        "present_readings": present_readings,
        "missing_readings": missing_readings,
        "missing_rate": missing_readings / possible_readings if possible_readings else 0.0,
        "normal_labels": int((labels == 0).sum()),
        "theft_labels": int((labels == 1).sum()),
        "missing_labels": int(labels.isna().sum()),
    }
