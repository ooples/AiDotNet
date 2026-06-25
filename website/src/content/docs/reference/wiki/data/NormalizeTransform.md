---
title: "NormalizeTransform<T>"
description: "Normalizes an array of values using mean and standard deviation: (x - mean) / std."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms.Numeric`

Normalizes an array of values using mean and standard deviation: (x - mean) / std.

## For Beginners

Normalization makes your data have zero mean and unit variance,
which helps neural networks train faster and more stably.

## How It Works

This is the standard normalization used in image preprocessing and feature scaling.
Each element is independently normalized using the provided per-channel or global mean/std.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NormalizeTransform(,,Int32)` | Creates a normalize transform with a single global mean and standard deviation. |
| `NormalizeTransform([],[])` | Creates a normalize transform with per-element mean and standard deviation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply([])` |  |

