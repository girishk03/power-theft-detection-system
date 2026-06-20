# Data Quality Report

## Scope

This report describes the committed file `data/raw/Electricity_Theft_Data.csv`. It records observable properties only; it does not assign an unverified source, license, or collection methodology.

## Completeness

| Measure | Verified value |
|---|---:|
| Customer rows | 9,957 |
| Daily reading columns | 365 |
| Period | January 1–December 31, 2015 |
| Possible readings | 3,634,305 |
| Present readings | 3,140,639 |
| Missing readings | 493,666 |
| Missing-reading rate | 13.58% |
| Duplicate customer identifiers | 0 |

Missingness is substantial and may not be random. Any future statistical or ML analysis must document per-customer missingness, imputation, exclusion rules, and sensitivity to those choices.

## Label Distribution

| `CHK_STATE` value | Interpretation | Rows | Share of non-missing labels |
|---|---|---:|---:|
| `0` | Normal | 8,562 | 86.00% |
| `1` | Theft | 1,394 | 14.00% |
| Missing | Unknown | 1 | Excluded from percentage |

The minority theft class requires stratified or customer-aware evaluation. Accuracy alone would be insufficient for future model reporting.

## Time Coverage

The included file contains 365 daily columns for **2015 only**. It does not support claims of measured data from 2014–2025. Dashboard years outside 2015 are simulated unless another compatible dataset is supplied.

## Runtime Sampling

`DATA_MODE=real` uses `pandas.read_csv(..., nrows=1000)`. The runtime therefore analyzes at most the first 1,000 customer rows, not all 9,957. This bounded load is tested explicitly.

## Schema Compatibility

The raw file uses `DD-MM-YY` date columns. The dashboard's current real-mode parser expects slash-delimited date columns containing a four-digit year. The file requires schema normalization before all year-specific endpoints can consume it directly.

## Source and License Warning

The repository does not contain an authoritative source URL, publisher record, license, or citation requirement for the CSV. Dataset source and redistribution rights remain **unverified**. The repository's MIT license applies to project code, not automatically to this dataset.

## Recommended Next Steps

1. Establish source, license, citation, and permitted-use records.
2. Normalize date columns with a tested conversion step.
3. Profile missingness by customer and date.
4. Define label-cleaning rules for the missing label.
5. Use customer-level data splits to prevent leakage.
6. Record transformations and data hashes for reproducibility.
