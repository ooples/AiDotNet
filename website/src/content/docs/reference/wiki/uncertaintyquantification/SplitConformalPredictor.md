---
title: "SplitConformalPredictor<T>"
description: "Implements Split Conformal Prediction for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.UncertaintyQuantification.ConformalPrediction`

Implements Split Conformal Prediction for regression tasks.

## For Beginners

Conformal Prediction is a framework that provides prediction intervals
with guaranteed coverage, regardless of the underlying model.

Key concepts:

- Instead of a single prediction, you get a prediction interval: [lower_bound, upper_bound]
- The interval is guaranteed to contain the true value with a specified probability (e.g., 90%)
- This guarantee holds for ANY model (neural network, random forest, etc.)

Example:
If you set confidence level = 90%:

- The model predicts: "House price will be between $180K and $220K"
- You're guaranteed that at least 90% of such intervals contain the true price

How it works:

1. Split data into training set, calibration set, and test set
2. Train your model on the training set
3. Use the calibration set to compute "non-conformity scores" (prediction errors)
4. Use these scores to build prediction intervals for new data

This is particularly valuable when you NEED reliability guarantees, such as:

- Medical diagnosis (must know uncertainty bounds)
- Safety-critical systems (must guarantee coverage)
- Scientific applications (need statistically valid uncertainty)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SplitConformalPredictor(IModel<Tensor<>,Tensor<>,ModelMetadata<>>)` | Initializes a new instance of the SplitConformalPredictor class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(Matrix<>,Vector<>)` | Calibrates the conformal predictor using a calibration dataset. |
| `ComputeAverageIntervalWidth(Matrix<>,Double)` | Computes the average interval width on a test set. |
| `ComputeQuantile(Vector<>,Double)` | Computes the quantile of calibration scores. |
| `EvaluateCoverage(Matrix<>,Vector<>,Double)` | Evaluates the empirical coverage of prediction intervals on a test set. |
| `PredictWithInterval(Tensor<>,Double)` | Predicts with a conformal prediction interval. |

