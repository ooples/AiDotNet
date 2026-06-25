---
title: "BatchBALD<T>"
description: "Implements BatchBALD for joint batch selection in active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements BatchBALD for joint batch selection in active learning.

## For Beginners

BatchBALD extends BALD to select batches of samples that are
jointly informative. Unlike naive greedy selection that picks individual high-BALD samples,
BatchBALD considers the joint mutual information to avoid redundant selections.

## How It Works

**Problem with naive BALD:** Selecting top-k samples by individual BALD scores
may result in redundant samples that provide similar information.

**Solution:** BatchBALD computes joint mutual information for candidate batches:
I(y₁, y₂, ..., yₖ; θ|x₁, x₂, ..., xₖ, D)

**Algorithm (Greedy Approximation):**

**Reference:** Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). "BatchBALD:
Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BatchBALD(Int32,Double,Int32)` | Initializes a new instance of the BatchBALD class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddDropoutNoise(Tensor<>,Double,Int32)` | Adds simulated dropout noise to predictions. |
| `ComputeEntropy(Vector<>)` | Computes entropy: H = -Σ p × log(p). |
| `ComputeIndividualBALDFromProbs(List<Vector<>>,Int32)` | Computes individual BALD score from precomputed MC probabilities. |
| `ComputeIndividualBALDScore(List<Tensor<>>,Int32,Int32)` | Computes individual BALD score from MC predictions. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeJointMI(List<List<Vector<>>>,Int32)` | Computes approximate joint mutual information for a batch. |
| `ComputeMarginalGain(Dictionary<Int32,List<Vector<>>>,List<Int32>,Int32,Int32)` | Computes the marginal gain in joint mutual information when adding a candidate. |
| `ComputePairwiseRedundancy(List<Vector<>>,List<Vector<>>,Int32)` | Computes pairwise redundancy between two samples based on prediction agreement. |
| `ExtractProbabilities(Tensor<>,Int32,Int32)` | Extracts probabilities for a single sample from batch predictions. |
| `GetMCPredictions(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` | Gets MC predictions by running multiple forward passes. |
| `GetSelectionStatistics` |  |
| `GetTopCandidates(Vector<>,Int32)` | Gets top candidates based on individual BALD scores. |
| `GreedyBatchSelection(List<Tensor<>>,Int32[],Int32,Int32)` | Performs greedy batch selection based on joint mutual information. |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

