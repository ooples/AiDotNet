---
title: "ProbabilityCalibrator<T>"
description: "Calibrates model outputs to produce reliable probability estimates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Calibration`

Calibrates model outputs to produce reliable probability estimates.

## For Beginners

Calibration is like adjusting a thermometer. Your model might
always say "90 degrees" when the true temperature is 85. Calibration learns this
systematic error and corrects for it.

**Available methods:**

- Platt Scaling: Fits a logistic curve - good for SVMs and many classifiers
- Isotonic Regression: Non-parametric curve fitting - more flexible but needs more data
- Temperature Scaling: Simple division - popular for neural networks
- Histogram Binning: Averages within bins - simple and interpretable
- Beta Calibration: Fits beta distribution - good for bounded outputs

**Example usage:**

1. Train your model and get predicted probabilities
2. Fit the calibrator on a held-out validation set
3. Use calibrator to transform all future predictions

## How It Works

Probability calibration transforms raw model scores into well-calibrated probabilities.
A well-calibrated model means: if you see 1000 predictions of "70% probability", about
700 of those should be positive outcomes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProbabilityCalibrator(ProbabilityCalibratorOptions)` | Initializes a new instance of the calibrator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ProbabilityCalibrationMethod` | Gets the calibration method being used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBrierScore(Vector<>,Vector<>)` | Computes the Brier Score. |
| `ComputeECE(Vector<>,Vector<>,Int32)` | Computes the Expected Calibration Error (ECE). |
| `ComputeMCE(Vector<>,Vector<>,Int32)` | Computes the Maximum Calibration Error (MCE). |
| `Fit(Vector<>,Vector<>)` | Fits the calibrator to predicted scores and true labels. |
| `FitTransform(Vector<>,Vector<>)` | Fits the calibrator and transforms in one step. |
| `GetBetaParameters` | Gets the beta calibration parameters (a, b, c). |
| `GetPlattParameters` | Gets the Platt scaling parameters (A, B). |
| `GetReliabilityDiagram(Vector<>,Vector<>,Int32)` | Gets a reliability diagram (calibration curve data). |
| `GetTemperature` | Gets the temperature parameter. |
| `Transform(Vector<>)` | Transforms predicted scores into calibrated probabilities. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_betaA` | Beta calibration parameters. |
| `_binEdges` | Histogram bin edges and probabilities. |
| `_isFitted` | Whether the calibrator has been fitted. |
| `_isotonicPoints` | Isotonic regression points. |
| `_options` | Configuration options. |
| `_plattA` | Platt scaling parameters (A and B in sigmoid(Ax + B)). |
| `_temperature` | Temperature parameter for temperature scaling. |
| `_vennCalibrationScores` | Stored calibration data for Venn-ABERS per-point dual isotonic fits. |

