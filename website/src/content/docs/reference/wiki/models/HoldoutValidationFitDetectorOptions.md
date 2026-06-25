---
title: "HoldoutValidationFitDetectorOptions"
description: "Configuration options for the Holdout Validation Fit Detector, which analyzes model performance on separate training and validation datasets to identify overfitting, underfitting, and other model quality issues."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Holdout Validation Fit Detector, which analyzes model performance
on separate training and validation datasets to identify overfitting, underfitting, and other
model quality issues.

## For Beginners

This detector helps you understand if your machine learning model is
learning properly by comparing how well it performs on data it has seen during training versus new
data it hasn't seen before.

Think of it like testing a student's understanding: if they can only answer questions they've seen
before but struggle with new questions on the same topic, they've memorized answers rather than
truly understanding the subject. Similarly, a good model should perform well not just on its training
data but also on new, unseen data.

This detector uses several thresholds to identify common problems:

- Overfitting: The model performs much better on training data than validation data (memorization)
- Underfitting: The model performs poorly on both training and validation data (not learning enough)
- High Variance: The model's performance varies significantly across different validation sets
- Good Fit: The model performs well on both training and validation data (proper learning)
- Stability: The model's performance is consistent across different validation sets

By detecting these issues early, you can adjust your model or training approach to get better results.

## How It Works

Holdout validation is a technique where a portion of the available data is "held out" from training
and used only for validation. By comparing model performance on training data versus this held-out
validation data, we can detect various issues like overfitting (performing much better on training
than validation data) or underfitting (performing poorly on both datasets).

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for confirming good fit based on the absolute performance on validation data. |
| `HighVarianceThreshold` | Gets or sets the threshold for detecting high variance based on the relative difference between multiple validation runs. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on the relative difference between training and validation performance. |
| `StabilityThreshold` | Gets or sets the threshold for confirming model stability based on the relative difference between multiple validation runs. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting based on the absolute performance on training data. |

