---
title: "GradientBoostingFitDetectorOptions"
description: "Configuration options for the Gradient Boosting Fit Detector, which analyzes model fit quality to detect overfitting in gradient boosting models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Gradient Boosting Fit Detector, which analyzes model fit quality
to detect overfitting in gradient boosting models.

## For Beginners

Think of this as a quality control tool specifically designed for gradient
boosting models (like XGBoost, LightGBM, or similar algorithms). These models are powerful but can easily
"memorize" your training data instead of learning general patterns. This detector helps you identify when
that's happening by comparing how well your model performs on the data it was trained on versus new data
it hasn't seen before.

It's like testing if someone truly understands a subject versus just memorizing answers to specific test
questions. If they score 95% on questions they've seen before but only 65% on new questions about the same
topic, they've probably memorized answers rather than understanding the subject. Similarly, this detector
helps identify when your model is "memorizing" rather than "learning," which would make it perform poorly
on new data in real-world applications.

## How It Works

The Gradient Boosting Fit Detector monitors the performance difference between training and validation
data to identify when a gradient boosting model is overfitting. Overfitting occurs when a model performs
significantly better on training data than on new, unseen data, indicating that it has memorized the
training examples rather than learning generalizable patterns.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for considering model fit as good based on the similarity of training and validation performance. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on the difference between training and validation performance. |
| `SevereOverfitThreshold` | Gets or sets the threshold for detecting severe overfitting based on the difference between training and validation performance. |

