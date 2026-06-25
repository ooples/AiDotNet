---
title: "TabularAugmentationSettings"
description: "Tabular-specific augmentation settings with industry-standard defaults."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Augmentation`

Tabular-specific augmentation settings with industry-standard defaults.

## For Beginners

These settings control how tabular (spreadsheet-like) data
is augmented. Useful for improving model generalization on structured data.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the feature dropout rate. |
| `EnableFeatureDropout` | Gets or sets whether feature dropout is enabled. |
| `EnableFeatureNoise` | Gets or sets whether feature noise is enabled. |
| `EnableMixUp` | Gets or sets whether MixUp for tabular data is enabled. |
| `EnableSmote` | Gets or sets whether SMOTE is enabled. |
| `MixUpAlpha` | Gets or sets the MixUp alpha parameter. |
| `NoiseStdDev` | Gets or sets the standard deviation of feature noise. |
| `SmoteK` | Gets or sets the number of nearest neighbors for SMOTE. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

