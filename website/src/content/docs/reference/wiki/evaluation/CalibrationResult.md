---
title: "CalibrationResult<T>"
description: "Results from calibration analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Results from calibration analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `Bins` | Calibration bins with predicted vs observed frequencies. |
| `BrierScore` | Brier Score: mean squared error of probability predictions. |
| `ExpectedCalibrationError` | Expected Calibration Error: weighted average of \|predicted - observed\| across bins. |
| `IsWellCalibrated` | Whether the classifier is considered well-calibrated (ECE < 0.05). |
| `MaximumCalibrationError` | Maximum Calibration Error: worst bin's \|predicted - observed\|. |
| `NumSamples` | Total number of samples analyzed. |

