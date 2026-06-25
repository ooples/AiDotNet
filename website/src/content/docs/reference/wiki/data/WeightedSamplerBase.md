---
title: "WeightedSamplerBase<T>"
description: "Base class for weighted samplers providing common weight-based functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Sampling`

Base class for weighted samplers providing common weight-based functionality.

## How It Works

WeightedSamplerBase provides common functionality for samplers that use per-sample
weights, including cumulative probability computation and weighted random selection.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightedSamplerBase(IEnumerable<>,Nullable<Int32>,Boolean,Nullable<Int32>)` | Initializes a new instance of the WeightedSamplerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` |  |
| `NumSamples` |  |
| `Replacement` |  |
| `Weights` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCumulativeProbabilities` | Computes cumulative probability distribution from weights. |
| `ComputeCumulativeProbabilitiesCore` | Core implementation of cumulative probability computation. |
| `SampleWeightedIndex` | Samples an index based on the cumulative probabilities using binary search. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CumulativeProbabilities` | Cumulative probabilities for weighted sampling. |
| `NumOps` | Numeric operations for type T. |
| `NumSamplesOverride` | Number of samples to draw per epoch. |
| `ReplacementEnabled` | Whether to sample with replacement. |
| `WeightsArray` | The weights for each sample. |

