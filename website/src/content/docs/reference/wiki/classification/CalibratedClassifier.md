---
title: "CalibratedClassifier<T>"
description: "Wrapper that adds probability calibration to any probabilistic classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Calibration`

Wrapper that adds probability calibration to any probabilistic classifier.

## For Beginners

Many classifiers give poor probability estimates:

- Random Forest tends to push probabilities towards 0.5
- SVM's margin-based scores aren't true probabilities
- Neural networks without proper training can be overconfident

This wrapper fixes that by learning to transform raw scores into
well-calibrated probabilities that match actual event frequencies.

Example: If the model says "80% probability", approximately 80% of
such predictions should actually be positive.

Usage:

## How It Works

CalibratedClassifier wraps any probabilistic classifier and applies post-hoc probability
calibration to improve the reliability of predicted probabilities.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CalibratedClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier. |
| `CalibratedClassifier(IProbabilisticClassifier<>,CalibratedClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new CalibratedClassifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseClassifier` | Gets the base classifier. |
| `CalibrationMethod` | Gets the calibration method. |
| `IsTrained` | Gets whether the model is trained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `CalibrateProb()` | Applies the fitted calibration to a single probability. |
| `Clone` |  |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateNewInstance` | Gets the model type. |
| `Deserialize(Byte[])` |  |
| `FitBetaCalibration(Vector<>,Vector<>)` | Fits beta calibration. |
| `FitCalibration(Matrix<>,Vector<>)` | Fits the calibration model to uncalibrated predictions. |
| `FitIsotonicRegression(Vector<>,Vector<>)` | Fits isotonic regression calibration using PAVA. |
| `FitPlattScaling(Vector<>,Vector<>)` | Fits Platt scaling (sigmoid calibration). |
| `FitTemperatureScaling(Vector<>,Vector<>)` | Fits temperature scaling. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InterpolateIsotonic()` | Interpolates isotonic regression mapping. |
| `PredictProbabilities(Matrix<>)` | Gets calibrated probability predictions. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `SigmoidT()` | Numerically stable sigmoid in type T. |
| `Train(Matrix<>,Vector<>)` | Trains the base classifier and fits calibration. |
| `TrainWithCrossValidation(Matrix<>,Vector<>)` | Trains with cross-validation to use all data for calibration. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseClassifier` | The base classifier being calibrated. |
| `_betaA` | Beta calibration parameters (a, b, c). |
| `_isTrained` | Whether the model has been trained. |
| `_isotonicMapping` | Isotonic regression mapping (sorted by probability). |
| `_options` | Configuration options. |
| `_plattA` | Platt scaling parameters (A, B) for sigmoid calibration. |
| `_random` | Random number generator. |
| `_temperature` | Temperature scaling parameter. |

