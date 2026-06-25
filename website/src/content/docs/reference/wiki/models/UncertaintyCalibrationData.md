---
title: "UncertaintyCalibrationData<TInput, TOutput>"
description: "Provides optional calibration data for uncertainty quantification features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Inputs`

Provides optional calibration data for uncertainty quantification features.

## For Beginners

Think of calibration data as a small "reality check" dataset.
The model is trained on training data, then calibration data is used to tune uncertainty-related behavior without overfitting.

## How It Works

Calibration data is used by certain uncertainty features that require a held-out dataset separate from training data.
Examples include conformal prediction and probability calibration.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UncertaintyCalibrationData(,Boolean,,Vector<Int32>)` | Initializes a new calibration data container. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasLabels` | Gets whether classification labels were provided. |
| `HasTargets` | Gets whether regression targets were provided. |
| `Labels` | Gets the calibration class labels (classification-style calibration). |
| `X` | Gets the calibration inputs. |
| `Y` | Gets the calibration targets (regression-style calibration). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForClassification(,Vector<Int32>)` | Creates calibration data for classification-style calibration (e.g., temperature scaling and conformal prediction sets). |
| `ForRegression(,)` | Creates calibration data for regression-style calibration (e.g., conformal prediction intervals). |

