---
title: "NoOutlierRemoval<T, TInput, TOutput>"
description: "A no-operation outlier removal implementation that preserves all data without modification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection`

A no-operation outlier removal implementation that preserves all data without modification.

## For Beginners

This class is used when you don't want to remove any outliers from your data.
It simply returns the original data unchanged, which is useful as a default option or when
you've already cleaned your data.

## How It Works

**When to use:**

- When your data has already been preprocessed for outliers
- When you want to compare model performance with and without outlier removal
- When your domain requires keeping all data points (e.g., fraud detection)

## Methods

| Method | Summary |
|:-----|:--------|
| `RemoveOutliers(,)` | Returns the input data unchanged without removing any outliers. |

