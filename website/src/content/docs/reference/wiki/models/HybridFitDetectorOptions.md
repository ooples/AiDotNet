---
title: "HybridFitDetectorOptions"
description: "Configuration options for the Hybrid Fit Detector, which combines multiple model evaluation techniques to provide a comprehensive assessment of model quality."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Hybrid Fit Detector, which combines multiple model evaluation techniques
to provide a comprehensive assessment of model quality.

## For Beginners

This detector is like having a team of experts evaluate your machine
learning model from different angles. Instead of relying on just one way to check if your model is
learning properly, it uses several different methods and combines their insights.

Think of it like a comprehensive health checkup that includes multiple tests (blood work, physical
exam, imaging, etc.) rather than just checking your temperature. By looking at your model from
multiple perspectives, the Hybrid Fit Detector can give you a more complete picture of how well
your model is learning and where it might be having problems.

The detector helps identify three common scenarios:

- Overfitting: Your model has "memorized" the training data but doesn't generalize well to new data
- Underfitting: Your model is too simple and isn't capturing important patterns in the data
- Good Fit: Your model has found the right balance, learning meaningful patterns that generalize well

This hybrid approach is especially useful when individual detection methods might give conflicting
signals or when you want extra confidence in your model quality assessment.

## How It Works

The Hybrid Fit Detector uses a combination of approaches to evaluate model fit, including comparing
training and validation performance, analyzing residuals, and examining model complexity. This
comprehensive approach provides a more robust assessment than any single method alone, helping to
identify overfitting, underfitting, and good fit conditions with greater confidence.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for confirming good fit based on a composite score from multiple evaluation methods. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on a composite score from multiple evaluation methods. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting based on a composite score from multiple evaluation methods. |

