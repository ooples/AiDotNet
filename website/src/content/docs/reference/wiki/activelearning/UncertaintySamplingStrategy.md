---
title: "UncertaintySamplingStrategy<T, TInput, TOutput>"
description: "Uncertainty sampling strategy for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Strategies.Uncertainty`

Uncertainty sampling strategy for active learning.

## For Beginners

Uncertainty sampling is the simplest and most popular
active learning strategy. It selects samples where the model is most uncertain
about its prediction.

## How It Works

**Uncertainty Measures:**

**When to Use:**

**Reference:** Lewis and Gale "A Sequential Algorithm for Training Text Classifiers" (1994)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UncertaintySamplingStrategy` | Initializes a new uncertainty sampling strategy with default entropy measure. |
| `UncertaintySamplingStrategy(UncertaintyMeasure,ActiveLearnerConfig<>)` | Initializes a new uncertainty sampling strategy with specified measure. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScores(IFullModel<,,>,IDataset<,,>)` |  |
| `ComputeUncertainty(IFullModel<,,>,)` |  |
| `GetPredictionProbabilities(IFullModel<,,>,)` |  |
| `Reset` |  |
| `SelectSamples(IFullModel<,,>,IDataset<,,>,Int32)` |  |
| `UpdateState(Int32[],[])` |  |

