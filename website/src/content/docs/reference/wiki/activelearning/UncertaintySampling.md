---
title: "UncertaintySampling<T>"
description: "Implements uncertainty sampling for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements uncertainty sampling for active learning.

## For Beginners

Uncertainty sampling is one of the simplest and most popular
active learning strategies. It selects samples where the model is least confident about
its predictions. The intuition is that uncertain samples are near the decision boundary
and provide the most information for learning.

## How It Works

**Uncertainty measures:**

**Reference:** Lewis and Gale, "A Sequential Algorithm for Training Text Classifiers" (1994).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UncertaintySampling(UncertaintySampling<>.UncertaintyMeasure)` | Initializes a new instance of the UncertaintySampling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEntropy(Vector<>)` | Computes entropy: -Σ p*log(p). |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeLeastConfidence(Vector<>)` | Computes least confidence: 1 - max(probabilities). |
| `ComputeMarginSampling(Vector<>)` | Computes margin sampling: 1 - (P(1st) - P(2nd)). |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ComputeUncertainty(Vector<>)` | Computes uncertainty for a single sample based on the configured measure. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both uncertainty and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

