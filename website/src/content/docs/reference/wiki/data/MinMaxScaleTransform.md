---
title: "MinMaxScaleTransform<T>"
description: "Scales values to a target range using min-max normalization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms.Numeric`

Scales values to a target range using min-max normalization.

## For Beginners

Min-max scaling squeezes your data into a specific range
(default [0, 1]). This is commonly used for pixel values or when your algorithm
requires bounded inputs.

## How It Works

The formula is: x_scaled = (x - min) / (max - min) * (targetMax - targetMin) + targetMin

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MinMaxScaleTransform(Double[],Double[],Double,Double)` | Creates a min-max scaler from pre-computed statistics. |
| `MinMaxScaleTransform([][],Double,Double)` | Creates a min-max scaler from a reference dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply([])` |  |

