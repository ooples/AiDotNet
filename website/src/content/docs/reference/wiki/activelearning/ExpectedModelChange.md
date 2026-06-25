---
title: "ExpectedModelChange<T>"
description: "Implements Expected Model Change (EMC) for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements Expected Model Change (EMC) for active learning.

## For Beginners

Expected Model Change selects samples that would cause the largest
change to the model's parameters if they were labeled and used for training. The intuition is
that samples which significantly change the model provide the most learning value.

## How It Works

**How it works:**

**Variants:**

**Reference:** Settles et al., "Multiple-Instance Active Learning" (2008). ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpectedModelChange(ExpectedModelChange<>.ChangeMetric)` | Initializes a new instance of the ExpectedModelChange class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEGL(Vector<>,Int32)` | Computes Expected Gradient Length (EGL). |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeExpectedChange(Vector<>,Int32)` | Computes the expected model change for a single sample. |
| `ComputeGradientVariance(Vector<>,Int32)` | Computes variance of gradient lengths across possible labels. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeMaxGradient(Vector<>,Int32)` | Computes maximum gradient length across all possible labels. |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both expected change and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

