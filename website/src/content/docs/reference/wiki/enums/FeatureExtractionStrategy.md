---
title: "FeatureExtractionStrategy"
description: "Defines strategies for extracting features from higher-dimensional tensors."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines strategies for extracting features from higher-dimensional tensors.

## For Beginners

This defines different ways to handle complex data.

When data has multiple values for each feature (like pixels in an image),
we need a strategy to condense these into a single value for analysis.
Different strategies work better for different types of data.

## Fields

| Field | Summary |
|:-----|:--------|
| `Flatten` | Uses the first element as a representative value. |
| `Max` | Uses the maximum value across all dimensions. |
| `Mean` | Uses the average value across all dimensions. |
| `WeightedSum` | Uses a weighted sum with configurable weights. |

