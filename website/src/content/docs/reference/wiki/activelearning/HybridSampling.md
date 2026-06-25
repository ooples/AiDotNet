---
title: "HybridSampling<T>"
description: "Implements Hybrid Sampling that combines multiple active learning strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements Hybrid Sampling that combines multiple active learning strategies.

## For Beginners

Hybrid sampling combines the benefits of multiple active learning
strategies. For example, uncertainty sampling might select many similar samples near the
decision boundary, while diversity sampling ensures good coverage. Combining them gets the
best of both worlds.

## How It Works

**How it works:**

**Common combinations:**

**Reference:** Settles, "Active Learning Literature Survey" (2009).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridSampling(IEnumerable<ValueTuple<IActiveLearningStrategy<>,Double>>,HybridSampling<>.CombinationMethod)` | Initializes a new instance of the HybridSampling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `Strategies` | Gets the strategies used in this hybrid sampler. |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineMaximum(List<Vector<>>,Int32)` | Combines scores using maximum (optimistic combination). |
| `CombineMinimum(List<Vector<>>,Int32)` | Combines scores using minimum (conservative combination). |
| `CombineProduct(List<Vector<>>,Int32)` | Combines scores using product (weighted geometric mean). |
| `CombineRankFusion(List<Vector<>>,Int32)` | Combines scores using rank fusion (Borda count). |
| `CombineWeightedSum(List<Vector<>>,Int32)` | Combines scores using weighted sum of normalized scores. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `CreateUncertaintyDiversity(Double,Double)` | Creates a default hybrid strategy combining uncertainty sampling and diversity sampling. |
| `GetSelectionStatistics` |  |
| `NormalizeScores(Vector<>)` | Normalizes scores to [0, 1] range using min-max normalization. |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

