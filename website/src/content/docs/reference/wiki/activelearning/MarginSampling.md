---
title: "MarginSampling<T>"
description: "Implements margin sampling for active learning sample selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements margin sampling for active learning sample selection.

## For Beginners

Margin sampling selects samples where the difference between
the top two predicted class probabilities is smallest. A small margin indicates the model
is uncertain between two classes, suggesting the sample is near the decision boundary.

## How It Works

**Formula:** margin = P(y₁|x) - P(y₂|x), where y₁ and y₂ are the most likely classes.

**Selection:** Select samples with smallest margins (transformed as 1 - margin for
consistency with other strategies where higher = more informative).

**Advantages:**

**Complexity:** O(n × c) where n=pool size, c=number of classes.

**Reference:** Settles, B. (2012). "Active Learning." Morgan & Claypool Publishers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MarginSampling` | Initializes a new instance of the MarginSampling class. |

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
| `ComputeMarginScore(Vector<>)` | Computes margin score: 1 - (P(1st) - P(2nd)). |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both margin score and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

