---
title: "CountEncoder<T>"
description: "Encodes categorical features using frequency counts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using frequency counts.

## For Beginners

Instead of creating multiple columns, frequency encoding
replaces each category with how often it appears:

- Category "common" appearing 1000 times → 1000 (or 0.5 if normalized)
- Category "rare" appearing 10 times → 10 (or 0.005 if normalized)

This is useful when the popularity of a category is predictive of the target.

## How It Works

CountEncoder replaces each category with its frequency count (number of occurrences)
in the training data. This creates a continuous feature that captures category popularity.

Options include normalizing counts to probabilities (0-1 range) or log-transforming
the counts to handle highly skewed distributions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CountEncoder(Boolean,Boolean,CountEncoderHandleUnknown,Double,Int32[])` | Creates a new instance of `CountEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CountMaps` | Gets the count maps for each column. |
| `HandleUnknown` | Gets how unknown categories are handled. |
| `LogTransform` | Gets whether counts are log-transformed. |
| `Normalize` | Gets whether counts are normalized to probabilities. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Learns the frequency counts from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for frequency encoding. |
| `TransformCore(Matrix<>)` | Transforms the data by replacing categories with their frequency counts. |

