# PR #431 - Uncertainty Quantification MVP TODO

This checklist tracks the remaining production tasks to reach the “exhaustive MVP” described in `docs/UncertaintyQuantification/PR431_GAP_ANALYSIS_AND_IMPLEMENTATION_PLAN.md`.

## Contract / Facade

- [x] Replace tuple return type with `UncertaintyPredictionResult<T, TOutput>` (no backwards-compat).
- [x] Ensure `PredictWithUncertainty` supports `Method.Auto` routing policy (conformal-first when calibration exists).
- [x] Ensure classification `Prediction` is calibrated probabilities when calibration is configured.
- [x] Ensure entropy + MI keys are always present in `UncertaintyPredictionResult.Metrics`.

## Builder-time calibration data (required)

- [x] Add a builder API to supply calibration data at build-time (separate from internal train/val/test split).
- [x] Validate calibration data shapes and task compatibility (regression vs classification).
- [x] Persist only the necessary calibration artifacts into `PredictionModelResult` (not raw calibration data).

## Conformal prediction (required)

- [x] Regression conformal:
  - [x] Calibrate residual quantile at build-time.
  - [x] Return `RegressionInterval` from `PredictWithUncertainty` when enabled.
- [x] Classification conformal:
  - [x] Calibrate threshold at build-time using class labels.
  - [x] Return `ClassificationSet` from `PredictWithUncertainty` when enabled.

## Calibration (temperature scaling) (recommended; required for "calibrated probabilities" decision)

- [x] Fit temperature at build-time (classification-like outputs).
- [x] Apply temperature scaling in `PredictWithUncertainty` output path.
- [ ] Report aggregate calibration metrics (e.g. ECE) via `DataSetStats.UncertaintyStats` where appropriate.

## Bayesian training (required)

- [x] Wire Bayes-by-Backprop training end-to-end (loss integration + KL regularization).
- [ ] Ensure serialization / metadata covers Bayesian parameters as needed.
- [x] Add tests for Bayesian training correctness and UQ outputs.

## Deep ensembles (required)

- [x] Define ensemble training policy (ensemble size, seed policy, cloning/training).
- [x] Implement end-to-end ensemble build + inference routing.
- [x] Add tests validating ensemble variance > 0 and stable shapes.

## Concurrency + determinism

- [x] Ensure `PredictionModelResult.PredictWithUncertainty` is safe under concurrent calls (serving).
- [x] Provide deterministic sampling when `RandomSeed` is set (per-call behavior).

## Tests

- [x] Regression: conformal interval returned and has correct ordering.
- [x] Classification: entropy+MI computed and conformal set returned; calibrated probabilities path is exercised.
- [x] Concurrency test for `PredictWithUncertainty` (no cross-call contamination).
