---
title: "ConfusionMatrixFitDetectorOptions"
description: "Configuration options for the Confusion Matrix Fit Detector, which evaluates how well a classification model performs."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Confusion Matrix Fit Detector, which evaluates how well a classification model performs.

## For Beginners

When you build a model that predicts categories (like "spam" vs "not spam" 
or "dog" vs "cat" vs "bird"), you need a way to check how good your model is. A confusion matrix is a table 
that shows how often your model was right or wrong for each category. This class provides settings that help 
automatically evaluate your model's performance based on that table. It can tell you if your model is doing 
well overall, if it's struggling with certain categories, or if your data is unbalanced (having way more 
examples of one category than others). Think of it like an automated grading system for your AI model.

## How It Works

The Confusion Matrix Fit Detector analyzes classification results to determine if a model is performing adequately.
It uses various thresholds and metrics to categorize model performance as good, moderate, or poor,
and can detect issues like class imbalance that might affect model reliability.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassImbalanceThreshold` | Gets or sets the threshold that determines when class imbalance is considered significant. |
| `ConfidenceThreshold` | Gets or sets the confidence threshold used for converting probability predictions to class labels. |
| `GoodFitThreshold` | Gets or sets the threshold above which a model's performance is considered good. |
| `ModerateFitThreshold` | Gets or sets the threshold above which a model's performance is considered moderate. |
| `PrimaryMetric` | Gets or sets the primary metric used to evaluate model performance. |

