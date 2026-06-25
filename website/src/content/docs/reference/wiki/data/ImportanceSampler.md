---
title: "ImportanceSampler<T>"
description: "A sampler that implements importance sampling for variance reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that implements importance sampling for variance reduction.

## For Beginners

Not all training samples are equally useful.
Importance sampling focuses training on the most informative samples:

- **High gradient norm** = Sample provides strong learning signal
- **High loss** = Model is uncertain, needs more training
- **High uncertainty** = Model needs to see this more

This can reduce training time by 2-3x compared to uniform sampling!

Example:

## How It Works

ImportanceSampler samples data points based on their importance, typically
computed from gradient norms, loss values, or uncertainty estimates.
This can accelerate training by focusing on samples that contribute most to learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImportanceSampler(Int32,Double,Boolean,Nullable<Int32>)` | Initializes a new instance of the ImportanceSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ImportanceScores` | Gets the importance scores for all samples. |
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetCorrectionFactor(Int32)` | Gets the sampling weight correction factor for a sample. |
| `GetIndicesCore` |  |
| `GetIndicesWithoutReplacement(Int32)` | Gets importance-weighted sample indices without replacement. |
| `RecomputeProbabilities` | Recomputes the cumulative probability distribution. |
| `SetImportances(IReadOnlyList<>)` | Sets all importance scores at once. |
| `UpdateImportance(Int32,)` | Updates the importance score for a single sample. |
| `UpdateImportances(IReadOnlyList<Int32>,IReadOnlyList<>)` | Batch updates importance scores. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

