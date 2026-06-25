---
title: "PrecisionRecallCurveFitDetectorOptions"
description: "Configuration options for the Precision-Recall Curve Fit Detector, which evaluates model quality using precision-recall metrics particularly valuable for imbalanced classification problems."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Precision-Recall Curve Fit Detector, which evaluates model quality
using precision-recall metrics particularly valuable for imbalanced classification problems.

## For Beginners

The Precision-Recall Curve Fit Detector helps evaluate how well your model is performing, especially when you have imbalanced data.

Imagine you're building a system to detect rare events (like fraud):

- You might have 1,000 normal transactions for every 1 fraudulent one
- A model that always predicts "not fraud" would be 99.9% accurate, but useless!
- This is why we need better ways to evaluate models with imbalanced data

Instead of simple accuracy, this detector uses two important metrics:

1. Precision: When the model predicts something is positive (like fraud), how often is it correct?
- High precision means fewer false alarms
- Think of it as: "When the model raises an alert, how trustworthy is that alert?"

2. Recall: Out of all the actual positive cases, how many did the model correctly identify?
- High recall means fewer missed cases
- Think of it as: "What percentage of fraudulent transactions did the model catch?"

The precision-recall curve shows the trade-off between these metrics at different thresholds.
This class lets you configure how the detector evaluates model quality based on these metrics.

## How It Works

The Precision-Recall Curve Fit Detector assesses model performance using metrics derived from the
precision-recall curve, which plots precision against recall at various classification thresholds.
Unlike accuracy, which can be misleading for imbalanced datasets, precision and recall metrics provide
more meaningful insights into model performance when class distributions are skewed. The Area Under
the Precision-Recall Curve (AUC-PR) and F1 Score are combined with customizable weights to produce
a composite fitness score. This detector is particularly valuable for applications where false positives
and false negatives have different implications, such as fraud detection, medical diagnosis, or anomaly
detection. The thresholds and weights configured in this class determine whether a model is considered
adequately fitted based on these metrics.

## Properties

| Property | Summary |
|:-----|:--------|
| `AreaUnderCurveThreshold` | Gets or sets the minimum acceptable Area Under the Precision-Recall Curve (AUC-PR) value. |
| `AucWeight` | Gets or sets the weight applied to the Area Under the Precision-Recall Curve in the composite fitness score. |
| `F1ScoreThreshold` | Gets or sets the minimum acceptable F1 Score for the model. |
| `F1ScoreWeight` | Gets or sets the weight applied to the F1 Score in the composite fitness score. |

