---
title: "DataAggregationHelper"
description: "Helper class for aggregating data samples."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Helper class for aggregating data samples.

## For Beginners

When loading data in batches or streaming,
you often need to combine multiple smaller pieces into one larger structure.
This helper provides optimized methods for this common operation.

## How It Works

DataAggregationHelper provides utility methods for combining multiple
data samples (Matrix, Vector, or Tensor) into a single aggregated structure.

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(List<>,String)` | Aggregates a list of data samples into a single structure. |
| `AggregateMatrices(List<Matrix<>>)` | Aggregates multiple matrices by concatenating rows. |
| `AggregateTensors(List<Tensor<>>)` | Aggregates multiple tensors by concatenating along the first dimension. |
| `AggregateVectors(List<Vector<>>)` | Aggregates multiple vectors by concatenating elements. |
| `CastToDataType()` | Casts a source type to a target type using implicit boxing. |

