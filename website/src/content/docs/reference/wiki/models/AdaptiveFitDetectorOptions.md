---
title: "AdaptiveFitDetectorOptions"
description: "Configuration options for the Adaptive Fit Detector, which automatically selects the most appropriate method to detect overfitting and underfitting in machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Adaptive Fit Detector, which automatically selects the most appropriate method
to detect overfitting and underfitting in machine learning models.

## For Beginners

Think of this as a smart diagnostic tool that checks if your AI model is learning properly.
Just like a doctor might use different tests depending on your symptoms, this detector chooses the right method to check
if your model is learning too much detail from your data (overfitting) or not learning enough (underfitting).
It automatically picks the best testing approach based on how complex your model is and how well it's performing.

## How It Works

The Adaptive Fit Detector combines multiple strategies to determine if a model is properly fitted to the data,
overfitted (too complex, memorizing training data), or underfitted (too simple, missing patterns).
It dynamically selects the most appropriate detection method based on the model's complexity and performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComplexityThreshold` | Gets or sets the threshold that determines when to switch between different fit detection methods based on model complexity. |
| `HybridOptions` | Gets or sets the configuration options for the Hybrid method of fit detection. |
| `LearningCurveOptions` | Gets or sets the configuration options for the Learning Curve method of fit detection. |
| `PerformanceThreshold` | Gets or sets the threshold that determines when to switch between different fit detection methods based on model performance. |
| `ResidualAnalysisOptions` | Gets or sets the configuration options for the Residual Analysis method of fit detection. |

