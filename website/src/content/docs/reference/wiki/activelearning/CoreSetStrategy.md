---
title: "CoreSetStrategy<T, TInput, TOutput>"
description: "CoreSet strategy for active learning using geometric diversity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Strategies.Diversity`

CoreSet strategy for active learning using geometric diversity.

## For Beginners

CoreSet is a diversity-based strategy that aims to select
samples that best represent the entire dataset. It's like choosing a small set of
"core" points that cover the data space well.

## How It Works

**How CoreSet Works:**

**Distance Metrics:**

**When to Use:**

**Reference:** Sener and Savarese "Active Learning for Convolutional Neural Networks: A Core-Set Approach" (ICLR 2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CoreSetStrategy` | Initializes a new CoreSet strategy with Euclidean distance. |
| `CoreSetStrategy(DistanceMetric,ActiveLearnerConfig<>)` | Initializes a new CoreSet strategy with specified distance metric. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDensityWeights(IDataset<,,>)` |  |
| `ComputeDistanceMatrix(IDataset<,,>)` |  |
| `ComputeDiversity(,)` |  |
| `ComputeDiversityScores(IDataset<,,>,IDataset<,,>)` |  |
| `ComputeScores(IFullModel<,,>,IDataset<,,>)` |  |
| `GetFeatureRepresentation()` |  |
| `InitializeLabeledFeatures(IDataset<,,>,IFullModel<,,>)` | Sets the initial labeled features for distance computation. |
| `Reset` |  |
| `SelectDiverseSamples(IDataset<,,>,IDataset<,,>,Int32)` |  |
| `SelectSamples(IFullModel<,,>,IDataset<,,>,Int32)` |  |
| `UpdateState(Int32[],[])` |  |

