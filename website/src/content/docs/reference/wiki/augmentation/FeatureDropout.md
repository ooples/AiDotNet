---
title: "FeatureDropout<T>"
description: "Randomly masks (sets to zero) features in tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Randomly masks (sets to zero) features in tabular data.

## For Beginners

Feature dropout randomly "hides" some features during training
by setting them to zero. This forces the model to learn more robust representations
that don't rely too heavily on any single feature.

## How It Works

**When to use:**

- To prevent overfitting to specific features
- To simulate missing data scenarios
- When you want the model to be robust to feature absence

**When NOT to use:**

- When all features are critical and cannot be missing
- When features have strong interdependencies that break when one is missing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureDropout(Double,Double,Int32[])` | Creates a new feature dropout augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropValue` | Gets or sets the value to use for dropped features. |
| `DropoutRate` | Gets the probability of dropping each feature. |
| `FeatureIndices` | Gets or sets the indices of features that can be dropped. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `GetParameters` |  |

