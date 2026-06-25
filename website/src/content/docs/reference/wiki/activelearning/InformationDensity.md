---
title: "InformationDensity<T>"
description: "Implements information density sampling for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements information density sampling for active learning.

## For Beginners

Information density measures how representative a sample is
of the overall data distribution by computing its average similarity to all other samples
in the pool. This helps select informative samples that are also typical of the data.

## How It Works

**Formula:** ID(x) = Uncertainty(x) × [1/|U| × Σ sim(x, x')]^β

where sim(x, x') is the similarity between samples (typically cosine or RBF kernel).

**Intuition:** A sample is information-dense if:

**Advantages:**

**Reference:** McCallum, A. & Nigam, K. (1998). "Employing EM and Pool-Based
Active Learning for Text Classification."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InformationDensity(Double,InformationDensity<>.SimilarityMeasure,Double)` | Initializes a new instance of the InformationDensity class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAverageSimilarities(Tensor<>,Int32,Int32)` | Computes average similarity of each sample to all other samples in the pool. |
| `ComputeCosineSimilarity(Tensor<>,Int32,Int32,Int32)` | Computes cosine similarity between two samples. |
| `ComputeEntropy(Vector<>)` | Computes entropy for uncertainty estimation. |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeEuclideanDistanceDouble(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance returning double. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeInverseEuclidean(Tensor<>,Int32,Int32,Int32)` | Computes inverse Euclidean distance similarity. |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ComputeRBFSimilarity(Tensor<>,Int32,Int32,Int32)` | Computes RBF (Gaussian) kernel similarity. |
| `ComputeSimilarity(Tensor<>,Int32,Int32,Int32)` | Computes similarity between two samples based on configured measure. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both information density and batch diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

