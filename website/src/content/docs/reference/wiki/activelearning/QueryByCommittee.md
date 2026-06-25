---
title: "QueryByCommittee<T>"
description: "Implements Query-by-Committee (QBC) for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements Query-by-Committee (QBC) for active learning.

## For Beginners

Query-by-Committee uses multiple models (a "committee") to evaluate
samples. It selects samples where the committee members disagree the most. The intuition is
that disagreement indicates uncertainty in the version space - the region of hypotheses
consistent with the labeled data.

## How It Works

**How it works:**

**Reference:** Seung et al., "Query by Committee" (1992). COLT.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryByCommittee(IEnumerable<IFullModel<,Tensor<>,Tensor<>>>,QueryByCommittee<>.DisagreementMeasure)` | Initializes a new instance of the QueryByCommittee class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Committee` | Gets the committee of models. |
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAverageKL(List<Tensor<>>,Int32,Int32)` | Computes average KL divergence between each member and the consensus. |
| `ComputeDisagreement(List<Tensor<>>,Int32,Int32)` | Computes disagreement for a single sample based on the configured measure. |
| `ComputeEuclideanDistance(Tensor<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `ComputeMinDistanceToSelected(Tensor<>,Int32,List<Int32>,Int32)` | Computes minimum distance from a sample to already selected samples. |
| `ComputePredictionVariance(List<Tensor<>>,Int32,Int32)` | Computes variance of predictions across committee members. |
| `ComputeVoteEntropy(List<Tensor<>>,Int32,Int32)` | Computes vote entropy: entropy of the vote distribution across committee. |
| `GetSelectionStatistics` |  |
| `GetSoftmaxProb(Tensor<>,Int32,Int32,Int32)` | Gets softmax probability for a specific class. |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `SelectTopScoring(Vector<>,Int32)` | Selects top-scoring samples. |
| `SelectWithDiversity(Vector<>,Tensor<>,Int32)` | Selects samples considering both committee disagreement score and diversity. |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

