---
title: "ProbabilityCalibratorOptions"
description: "Configuration options for probability calibrator."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for probability calibrator.

## For Beginners

Many models give probability-like outputs, but these aren't
always true probabilities. For example:

- A decision tree might output 0.9 for all "yes" predictions (overconfident)
- A neural network might output values between 0.3-0.7 only (underconfident)
- SVM outputs are not probabilities at all without calibration

Calibration transforms these outputs into reliable probabilities:

- If you see 1000 predictions of "60% chance", about 600 should be correct
- This is crucial for decision-making (medical diagnoses, financial risk, etc.)

**When to use:**

- Whenever you need actual probability estimates (not just rankings)
- When combining predictions from different models
- For threshold-based decisions where probability values matter

## How It Works

Probability calibration ensures that predicted probabilities are reliable. When a
calibrated model says "70% chance", you should expect the event to occur about 70%
of the time across many such predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `CalibratorMethod` | Gets or sets the calibration method. |
| `LearningRate` | Gets or sets the learning rate for gradient-based methods. |
| `MaxIterations` | Gets or sets the maximum number of iterations for optimization-based methods. |
| `NumBins` | Gets or sets the number of bins for histogram-based methods. |
| `NumFolds` | Gets or sets the number of cross-validation folds. |
| `Regularization` | Gets or sets the regularization strength. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `Tolerance` | Gets or sets the convergence tolerance. |
| `UseCrossValidation` | Gets or sets whether to use cross-validation for calibration. |

