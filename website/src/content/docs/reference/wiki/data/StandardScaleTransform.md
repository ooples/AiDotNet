---
title: "StandardScaleTransform<T>"
description: "Applies Z-score normalization: (x - mean) / std, computing mean and std from a reference dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms.Numeric`

Applies Z-score normalization: (x - mean) / std, computing mean and std from a reference dataset.

## For Beginners

Z-score normalization converts your data so that
the average becomes 0 and the spread becomes 1. This is computed from a reference
dataset (typically the training set) and then applied to all data.

## How It Works

Unlike `NormalizeTransform` which requires pre-computed mean/std,
this transform computes statistics from a reference dataset during construction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StandardScaleTransform(Double[],Double[])` | Creates a standard scaler from pre-computed statistics. |
| `StandardScaleTransform([][])` | Creates a standard scaler from a reference dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply([])` |  |

