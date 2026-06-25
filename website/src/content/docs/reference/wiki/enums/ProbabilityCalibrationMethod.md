---
title: "ProbabilityCalibrationMethod"
description: "Defines probability calibration strategies for classification-like outputs."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines probability calibration strategies for classification-like outputs.

## For Beginners

Calibration helps ensure that "80% confident" means "correct about 80% of the time".

## How It Works

Calibration transforms predicted probabilities to better reflect empirical correctness likelihoods.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically selects a suitable calibration method based on the task and output shape. |
| `BayesianBinning` | Bayesian Binning into Quantiles (BBQ) - adaptive binning. |
| `BetaCalibration` | Uses beta calibration (more flexible than Platt scaling, handles asymmetric distortions). |
| `HistogramBinning` | Histogram Binning - assigns average probability to each bin. |
| `IsotonicRegression` | Uses isotonic regression calibration (non-parametric monotonic calibration, typically for binary classification). |
| `None` | Disables probability calibration. |
| `PlattScaling` | Uses Platt scaling (logistic calibration, typically best for binary classification). |
| `TemperatureScaling` | Uses temperature scaling (typically best for multiclass neural network probabilities/logits). |
| `VennABERS` | Venn-ABERS - provides probability intervals, not point estimates. |

