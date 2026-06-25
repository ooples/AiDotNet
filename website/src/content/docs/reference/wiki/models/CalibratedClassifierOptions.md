---
title: "CalibratedClassifierOptions<T>"
description: "Configuration options for calibrated classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for calibrated classifiers.

## For Beginners

Calibration improves the probability estimates from classifiers.

Choose your method based on your needs:

- **Platt Scaling**: Good for SVMs and linear models. Assumes S-shaped transformation.
- **Isotonic Regression**: Non-parametric, very flexible. Best with lots of data.
- **Beta Calibration**: More flexible than Platt, handles asymmetric distortions.
- **Temperature Scaling**: Simple and effective for neural networks.

Key settings:

- Use cross-validation (CrossValidationFolds > 1) for better calibration with less data
- More CV folds = better calibration but slower training
- Set Seed for reproducibility

## Properties

| Property | Summary |
|:-----|:--------|
| `CalibrationMethod` | Gets or sets the calibration method. |
| `CalibrationSetFraction` | Gets or sets the fraction of data to use for calibration when not using cross-validation. |
| `CrossValidationFolds` | Gets or sets the number of cross-validation folds. |

