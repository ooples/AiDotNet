---
title: "BALD<T>"
description: "Implements Bayesian Active Learning by Disagreement (BALD) for sample selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements Bayesian Active Learning by Disagreement (BALD) for sample selection.

## For Beginners

BALD uses information theory to select samples that maximize
the mutual information between model predictions and model parameters. In practice, this
means selecting samples where different "versions" of the model disagree the most.

## How It Works

**Formula:** I(y; θ|x, D) = H(y|x, D) - E_θ[H(y|x, θ)]

where H is entropy, y is the label, θ are model parameters, x is input, D is training data.

**Interpretation:**

**Implementation:** Uses MC Dropout to approximate Bayesian inference by running
multiple forward passes with dropout enabled during inference.

**Reference:** Houlsby, N. et al. (2011). "Bayesian Active Learning for Classification
and Preference Learning."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BALD(Int32,Double)` | Initializes a new instance of the BALD class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddDropoutNoise(Tensor<>,Double,Int32)` | Adds simulated dropout noise to predictions. |
| `ComputeBALDScore(List<Tensor<>>,Int32,Int32)` | Computes BALD score: I(y; θ\|x) = H(y\|x) - E[H(y\|x, θ)]. |
| `ComputeEntropy(Vector<>)` | Computes entropy: H = -Σ p × log(p). |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples without diversity consideration. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both BALD score and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

