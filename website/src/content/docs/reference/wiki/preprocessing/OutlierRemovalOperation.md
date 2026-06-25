---
title: "OutlierRemovalOperation<T>"
description: "A row operation that removes or handles outliers using any anomaly detector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation`

A row operation that removes or handles outliers using any anomaly detector.

## For Beginners

Outliers are unusual data points that don't follow the pattern
of most of your data. They can confuse machine learning models and lead to poor
predictions. This operation identifies outliers using statistical methods and either:

- Removes them entirely (reduces your dataset size)
- Replaces their values with typical values (median or mean)

## How It Works

This operation wraps any `IAnomalyDetector` to identify outliers and
either remove them or replace their values. When using Remove mode, both features (X)
and labels (y) are modified together to maintain alignment.

**When to Use Each Mode:**

- **Remove:** When you have plenty of data and outliers are likely errors
- **ReplaceWithMedian:** When outliers are extreme but you need to preserve sample count
- **ReplaceWithMean:** Similar to median, but more affected by other outliers

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OutlierRemovalOperation(IAnomalyDetector<>,OutlierHandlingMode)` | Creates a new outlier removal operation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Detector` | Gets the underlying anomaly detector. |
| `IsFitted` |  |
| `Mode` | Gets the outlier handling mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitResample(Matrix<>,Vector<>)` |  |
| `FitResampleTensor(Tensor<>,Tensor<>)` |  |
| `GetOutlierMask(Matrix<>)` | Gets a mask indicating which rows were identified as outliers. |

