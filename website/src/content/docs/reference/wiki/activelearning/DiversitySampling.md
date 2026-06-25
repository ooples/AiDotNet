---
title: "DiversitySampling<T>"
description: "Implements Diversity Sampling for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements Diversity Sampling for active learning.

## For Beginners

Diversity Sampling selects samples that are representative of
different regions in the input space. Instead of focusing on uncertain samples near the
decision boundary, diversity sampling ensures good coverage of the data distribution.

## How It Works

**How it works:**

**Strategies:**

**Reference:** Sener and Savarese, "Active Learning for Convolutional Neural Networks:
A Core-Set Approach" (2018). ICLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiversitySampling(DiversitySampling<>.DiversityMethod,DiversitySampling<>.DistanceMetric)` | Initializes a new instance of the DiversitySampling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CoverageRadius` | Gets the coverage radius from the last selection. |
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAverageNearestNeighborDistance(Tensor<>,Int32,Int32,Int32)` | Computes the average k-nearest neighbor distance. |
| `ComputeCentroid(Tensor<>,Int32,Int32)` | Computes the centroid of the data. |
| `ComputeCosineDistance(Tensor<>,Int32,Int32,Int32)` | Computes cosine distance (1 - cosine similarity) between two samples. |
| `ComputeCoverageRadius(Tensor<>,Int32[])` | Computes the coverage radius for selected samples. |
| `ComputeDeltaDistances(Tensor<>,Vector<>,Int32,Int32)` | Computes delta distance (distance to nearest higher-density sample). |
| `ComputeDensityScores(Tensor<>,Int32,Int32)` | Computes density-based scores (inverse of average distance = high density). |
| `ComputeDistance(Tensor<>,Int32,Int32,Int32)` | Computes distance between two samples based on the configured metric. |
| `ComputeDistanceScores(Tensor<>,Int32,Int32)` | Computes distance-based scores (average distance to k nearest neighbors). |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeLocalDensities(Tensor<>,Int32,Int32)` | Computes local density for each sample using a Gaussian kernel. |
| `ComputeManhattanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Manhattan (L1) distance between two samples. |
| `FindNearestToCentroid(Tensor<>,Vector<>,Int32,Int32)` | Finds the sample nearest to the centroid. |
| `GetSelectionStatistics` |  |
| `SelectDensityPeaks(Tensor<>,Int32)` | Selects samples based on density peaks. |
| `SelectFarthestFirst(Tensor<>,Int32)` | Selects samples using farthest-first traversal. |
| `SelectKCenterGreedy(Tensor<>,Int32)` | Selects samples using k-center greedy algorithm for core-set construction. |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

