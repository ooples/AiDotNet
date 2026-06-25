---
title: "CoreSetSelection<T>"
description: "Implements core-set selection using the k-center-greedy algorithm for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements core-set selection using the k-center-greedy algorithm for active learning.

## For Beginners

Core-set selection aims to select samples that best represent
the overall data distribution. It uses the k-center algorithm which iteratively selects
the sample that is farthest from all previously selected samples. This ensures good
coverage of the feature space.

## How It Works

**Algorithm (k-center-greedy):**

**Advantages:**

**Complexity:** O(n × k × d) where n=pool size, k=batch size, d=feature dimension.

**Reference:** Sener & Savarese, "Active Learning for Convolutional Neural Networks:
A Core-Set Approach" (ICLR 2018).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CoreSetSelection(CoreSetSelection<>.DistanceMetric)` | Initializes a new instance of the CoreSetSelection class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCosineDistance(Tensor<>,Int32,Int32,Int32)` | Computes cosine distance between two samples: 1 - cosine_similarity. |
| `ComputeDataCenter(Tensor<>,Int32,Int32)` | Computes the center of the data. |
| `ComputeDistance(Tensor<>,Int32,Int32,Int32)` | Computes distance between two samples based on the configured metric. |
| `ComputeDistanceToCenter(Tensor<>,Int32,Vector<>,Int32)` | Computes distance from a sample to the data center. |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeManhattanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Manhattan distance between two samples. |
| `FindFarthestFromOrigin(Tensor<>,Int32)` | Finds the sample farthest from the origin. |
| `GetSelectionStatistics` |  |
| `KCenterGreedy(Tensor<>,Int32,Int32)` | Implements the k-center-greedy algorithm. |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

