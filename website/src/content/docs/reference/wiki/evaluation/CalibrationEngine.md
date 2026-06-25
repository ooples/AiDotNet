---
title: "CalibrationEngine<T>"
description: "Engine for analyzing and improving probability calibration of classifiers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Engine for analyzing and improving probability calibration of classifiers.

## For Beginners

A well-calibrated classifier produces probabilities that match actual outcomes:

- If model says 70% probability, ~70% of such predictions should be positive
- Many classifiers are NOT well-calibrated (e.g., Random Forest, SVM)
- Calibration methods: Platt scaling (sigmoid), Isotonic regression

## How It Works

**Why calibration matters:**

- Decision making: "Should I act if P > 0.8?" requires calibrated probabilities
- Ranking: AUC doesn't need calibration, but probability thresholds do
- Ensemble: Combining models requires comparable probability scales

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CalibrationEngine(CalibrationOptions)` | Initializes the calibration engine. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Analyze([],[],Int32)` | Analyzes the calibration of a classifier. |
| `IsotonicCalibration([],[])` | Applies isotonic regression calibration. |
| `PlattScaling([],[])` | Applies Platt scaling (sigmoid calibration) to probabilities. |

