---
title: "EntropySampling<T>"
description: "Implements entropy sampling for active learning sample selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements entropy sampling for active learning sample selection.

## For Beginners

Entropy sampling selects samples where the prediction entropy
is highest. High entropy means the probability distribution is spread out across classes,
indicating the model is uncertain about which class the sample belongs to.

## How It Works

**Formula:** H(y|x) = -Σ P(yᵢ|x) × log P(yᵢ|x)

**Interpretation:**

**Advantages:**

**Complexity:** O(n × c) where n=pool size, c=number of classes.

**Reference:** Shannon, C.E. (1948). "A Mathematical Theory of Communication."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EntropySampling` | Initializes a new instance of the EntropySampling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEntropy(Vector<>)` | Computes entropy: H = -Σ p × log(p). |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both entropy and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

