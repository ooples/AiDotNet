---
title: "OutputDistribution"
description: "Specifies the target distribution for quantile transformation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the target distribution for quantile transformation.

## For Beginners

Think of this as choosing the shape you want your data to take:

- Uniform: Spreads values evenly across the range (like a flat distribution)
- Normal: Creates a bell curve pattern (most values in the middle, fewer at extremes)

## How It Works

This enum defines the available output distributions for the QuantileTransformer.
Each distribution has different characteristics and use cases in machine learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `Normal` | Maps data to a normal (Gaussian) distribution with mean 0 and standard deviation 1. |
| `Uniform` | Maps data to a uniform distribution where all values are equally likely. |

