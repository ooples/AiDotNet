---
title: "CalibratedProbabilityFitDetectorOptions"
description: "Configuration options for the Calibrated Probability Fit Detector, which evaluates how well a model's  predicted probabilities match actual outcomes."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Calibrated Probability Fit Detector, which evaluates how well a model's 
predicted probabilities match actual outcomes.

## For Beginners

This class contains settings for a tool that checks if your AI model's confidence levels 
are trustworthy. Imagine a weather forecaster who predicts a 70% chance of rain - if it actually rains on 70% of the 
days when they make this prediction, they're well-calibrated. Similarly, we want AI models that say "I'm 90% sure" 
to be right about 90% of the time. This detector helps identify if your model is overconfident (saying it's very sure 
when it shouldn't be) or underconfident (not expressing enough confidence when it should).

## How It Works

Probability calibration measures whether a model's confidence (predicted probability) aligns with its actual accuracy.
For example, when a well-calibrated model predicts events with 80% confidence, those events should occur about 80% of the time.
This detector helps identify models that are overconfident or underconfident in their predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the maximum calibration error threshold for a model to be considered well-calibrated. |
| `MaxCalibrationError` | Gets or sets the maximum allowed calibration error before the detector reports a critical issue. |
| `NumCalibrationBins` | Gets or sets the number of bins used to group predictions for calibration assessment. |
| `OverfitThreshold` | Gets or sets the calibration error threshold above which a model is considered poorly calibrated. |

