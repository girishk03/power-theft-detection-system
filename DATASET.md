# Dataset Status and Quality Report

## Included Raw Dataset

- **Path:** `data/raw/Electricity_Theft_Data.csv`
- **Customer rows:** 9,957
- **Daily reading columns:** 365
- **Period:** January 1–December 31, 2015
- **Possible daily readings:** 3,634,305
- **Present readings:** 3,140,639
- **Missing readings:** 493,666
- **Duplicate customer identifiers:** 0

## Labels

The `CHK_STATE` column contains:

| Label | Meaning | Rows |
|---|---|---:|
| `0` | Normal | 8,562 |
| `1` | Theft | 1,394 |
| Missing | Unknown | 1 |

The labeled theft rate among non-missing labels is approximately 14%.

## Source and License

The repository history describes this file generically as electricity-theft data, but does not include:

- an authoritative source URL;
- an original dataset title or publisher record;
- collection methodology;
- license terms; or
- citation requirements.

Source and license status are therefore **unverified**. Do not assume that the project MIT license applies to this CSV. Confirm redistribution and usage rights before research, commercial, or public deployment use.

## Schema

- `CONS_NO`: customer identifier
- `DD-MM-YY` columns: daily consumption values for 2015
- `CHK_STATE`: binary normal/theft label

The file contains 493,666 missing readings, approximately 13.6% of all possible daily values. Any future model work should document imputation, leakage prevention, customer-level train/test splitting, and treatment of the missing label.

## Runtime Use

The dashboard defaults to `DATA_MODE=simulated` and does not load this file.

With `DATA_MODE=real`, `app.py` reads at most the first 1,000 rows from `DATASET_PATH`. The current year parser expects slash-delimited date columns, while this file uses hyphen-delimited `DD-MM-YY` columns. As committed, the CSV is not a fully integrated 2015–2025 runtime source.

When real data quality is insufficient, some endpoints generate illustrative consumption patterns. Responses must therefore not be presented as measured utility outcomes without additional provenance fields.

## Illustrative Sample

`data/sample_data.csv` contains 96 rows and five columns. It has 48 normal and 48 theft labels and no missing values. Its source and generation process are not documented, so it should be treated only as an illustrative schema example.

## Required Work Before ML Claims

1. Verify the original source, license, and citation.
2. Normalize and validate date columns.
3. Define customer-level train, validation, and test splits.
4. Publish a reproducible preprocessing and training command.
5. Save evaluation predictions and metric-generation code.
6. Report uncertainty, class-specific errors, and operational false-positive costs.
