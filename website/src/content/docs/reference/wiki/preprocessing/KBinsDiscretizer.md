---
title: "KBinsDiscretizer<T>"
description: "Bins continuous features into discrete intervals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Discretizers`

Bins continuous features into discrete intervals.

## For Beginners

This transformer groups continuous values into bins (categories):

- Uniform strategy: Divides the range into equal-width intervals
- Quantile strategy: Divides so each bin has roughly equal number of samples

Example with 3 bins using uniform strategy:
[1, 5, 10, 15, 20, 25] with range 1-25 creates bins:

- Bin 0: 1-9 → values 1, 5
- Bin 1: 9-17 → values 10, 15
- Bin 2: 17-25 → values 20, 25

## How It Works

KBinsDiscretizer discretizes features into k equal-width or quantile-based bins.
This is useful for transforming continuous features into categorical features,
which can help certain models like decision trees and reduce sensitivity to outliers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KBinsDiscretizer(Int32,BinningStrategy,EncodeMode,Int32[])` | Creates a new instance of `KBinsDiscretizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BinEdges` | Gets the bin edges for each feature computed during fitting. |
| `Encode` | Gets the encoding mode used. |
| `NBins` | Gets the number of bins used for discretization. |
| `Strategy` | Gets the binning strategy used. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the bin edges for each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reverses the discretization by returning bin midpoints. |
| `TransformCore(Matrix<>)` | Transforms the data by discretizing each feature into bins. |

