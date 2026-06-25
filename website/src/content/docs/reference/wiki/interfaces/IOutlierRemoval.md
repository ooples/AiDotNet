---
title: "IOutlierRemoval<T, TInput, TOutput>"
description: "Defines methods for detecting and removing outliers from datasets."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines methods for detecting and removing outliers from datasets.

## How It Works

**For Beginners:** Outliers are unusual data points that differ significantly from most of your data.
These unusual values can negatively impact machine learning models by skewing results.

This interface provides a standard way to implement different outlier detection and removal
techniques. By removing outliers, you can often improve the accuracy and reliability of your
machine learning models.

## Methods

| Method | Summary |
|:-----|:--------|
| `RemoveOutliers(,)` | Removes outliers from the input data and returns the cleaned dataset. |

