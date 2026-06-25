---
title: "CalibrationOptions"
description: "Configuration options for probability calibration analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for probability calibration analysis.

## For Beginners

When your model says "80% chance this email is spam", calibration
checks if it's really spam 80% of the time. Poorly calibrated models might say 80% but
actually be right 95% or only 50% of the time. Good calibration is essential when you
use predicted probabilities for decisions.

## How It Works

Calibration measures whether predicted probabilities match actual frequencies.
A well-calibrated model that predicts 80% probability should be correct 80% of the time.

## Properties

| Property | Summary |
|:-----|:--------|
| `BinningStrategy` | Binning strategy. |
| `BootstrapSamples` | Number of bootstrap samples for CIs. |
| `CVFolds` | Number of CV folds if using cross-validation. |
| `CalibrationMethod` | Calibration method to apply. |
| `ComputeAdaptiveECE` | Whether to compute Adaptive ECE. |
| `ComputeBrierDecomposition` | Whether to compute Brier score decomposition. |
| `ComputeCalibrationSlope` | Whether to compute calibration slope/intercept. |
| `ComputeConfidenceIntervals` | Whether to compute confidence intervals for calibration metrics. |
| `ComputeECE` | Whether to compute Expected Calibration Error (ECE). |
| `ComputeMCE` | Whether to compute Maximum Calibration Error (MCE). |
| `ComputePerClassCalibration` | Whether to compute per-class calibration. |
| `ConfidenceLevel` | Confidence level. |
| `GenerateConfidenceHistogram` | Whether to generate confidence histogram data. |
| `GenerateReliabilityDiagram` | Whether to generate reliability diagram data. |
| `MaxIterations` | Maximum iterations for calibrator optimization. |
| `NumberOfBins` | Number of bins for reliability diagram. |
| `RegularizationStrength` | Regularization strength for calibrator fitting. |
| `RunHosmerLemeshowTest` | Whether to run Hosmer-Lemeshow test. |
| `RunSpiegelhalterTest` | Whether to run Spiegelhalter's z-test. |
| `Temperature` | Temperature for temperature scaling calibration. |
| `UseCrossValidation` | Whether to use cross-validation for calibration metrics. |

