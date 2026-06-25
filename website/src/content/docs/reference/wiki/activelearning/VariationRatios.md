---
title: "VariationRatios<T>"
description: "Implements variation ratios for active learning sample selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements variation ratios for active learning sample selection.

## For Beginners

Variation ratios measure the proportion of predicted labels
that are NOT in the modal (most likely) class. High variation ratio indicates high
uncertainty because the model is not strongly predicting any single class.

## How It Works

**Formula:** VR(x) = 1 - max(P(y|x))

**Interpretation:**

**Advantages:**

**Reference:** Freeman, L.C. (1965). "Elementary Applied Statistics: For Students
in Behavioral Science."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationRatios` | Initializes a new instance of the VariationRatios class. |

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
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ComputeVariationRatio(Vector<>)` | Computes variation ratio: VR = 1 - max(P(y\|x)). |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both variation ratio and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

