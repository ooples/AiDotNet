---
title: "DensityWeightedSampling<T>"
description: "Implements density-weighted sampling for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements density-weighted sampling for active learning.

## For Beginners

Density-weighted sampling combines uncertainty with density
weighting to avoid selecting outliers. Even if a sample is uncertain, it may not be
informative if it's an outlier that doesn't represent the data distribution.

## How It Works

**Formula:** Score(x) = Uncertainty(x) × Density(x)^β

where Density(x) is computed using average distance to k nearest neighbors.

**Parameters:**

**Advantages:**

**Reference:** Settles, B. & Craven, M. (2008). "An Analysis of Active Learning
Strategies for Sequence Labeling Tasks."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DensityWeightedSampling(Double,Int32)` | Initializes a new instance of the DensityWeightedSampling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDensityScores(Tensor<>,Int32,Int32)` | Computes density scores based on average distance to k nearest neighbors. |
| `ComputeEntropy(Vector<>)` | Computes entropy for uncertainty estimation. |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `NormalizeDensityScores(Vector<>)` | Normalizes density scores to [0, 1] range. |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both density-weighted scores and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

