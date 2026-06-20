# Legacy Model Claims — Verification Status

## Status: Unverified and Not Reproducible

Earlier versions of this file reported Random Forest metrics, model sizes, training duration, feature counts, production readiness, and deployment outcomes. Those claims are not supported by the current repository because it does not include:

- a reproducible training entry point;
- saved model or scaler artifacts;
- train/test split records;
- evaluation predictions;
- confusion matrices, ROC data, or generated plots; or
- an experiment environment lockfile.

The historical values have been removed to prevent them from being interpreted as verified results. They must not be used in resumes, presentations, papers, or product claims unless a future reproducible experiment independently produces and records them.

## Verified Runtime Capability

The running Flask application uses the rule-based helper in `src/risk_scoring.py`. It demonstrates investigation prioritization with simulated or bounded CSV-backed inputs. It does not load an ML model.

## Requirements for Future Evaluation

A future result report should include:

1. dataset source and license;
2. immutable dataset and code hashes;
3. customer-level split methodology;
4. preprocessing and leakage controls;
5. exact training command and dependency versions;
6. saved predictions and metric-generation code;
7. per-class precision, recall, F1, and confusion matrix; and
8. an operational discussion of false-positive and false-negative costs.
