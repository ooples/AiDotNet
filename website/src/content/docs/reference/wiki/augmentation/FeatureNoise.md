---
title: "FeatureNoise<T>"
description: "Adds Gaussian noise to numerical features in tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Adds Gaussian noise to numerical features in tabular data.

## For Beginners

This augmentation adds small random variations to your data,
similar to measurement noise in real-world data. This helps models become robust to
small fluctuations and prevents overfitting to exact values.

## How It Works

**When to use:**

- Numerical features that have natural variation
- Small datasets where regularization is needed
- When you want to simulate measurement uncertainty

**When NOT to use:**

- Categorical features (use other augmentations)
- Features with strict constraints (e.g., binary flags)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureNoise(Double,Double,Int32[])` | Creates a new feature noise augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureIndices` | Gets or sets the indices of features to apply noise to. |
| `NoiseStdDev` | Gets the standard deviation of the noise. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `GetParameters` |  |

