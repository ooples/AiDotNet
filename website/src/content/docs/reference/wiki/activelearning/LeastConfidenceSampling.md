---
title: "LeastConfidenceSampling<T>"
description: "Implements least confidence sampling for active learning sample selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements least confidence sampling for active learning sample selection.

## For Beginners

Least confidence sampling selects samples where the model's
top prediction has the lowest probability. This is the simplest uncertainty measure,
focusing on how confident the model is in its best guess.

## How It Works

**Formula:** LC(x) = 1 - max P(y|x)

**Interpretation:**

**Advantages:**

**Complexity:** O(n × c) where n=pool size, c=number of classes.

**Reference:** Lewis, D.D. & Catlett, J. (1994). "Heterogeneous uncertainty
sampling for supervised learning."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeastConfidenceSampling` | Initializes a new instance of the LeastConfidenceSampling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeLeastConfidence(Vector<>)` | Computes least confidence score: LC = 1 - max(p). |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both least confidence and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

