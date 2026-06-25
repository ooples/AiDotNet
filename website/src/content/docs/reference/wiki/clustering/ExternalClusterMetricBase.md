---
title: "ExternalClusterMetricBase<T>"
description: "Base class for external cluster evaluation metrics that compare against ground truth."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Clustering.Evaluation`

Base class for external cluster evaluation metrics that compare against ground truth.

## For Beginners

This base class handles the common math that all
external metrics need:

- Building contingency tables (cross-tabulation of true vs predicted)
- Computing entropy (how "mixed" or "uncertain" clusters are)
- Counting pairs (how many point-pairs agree/disagree)

Individual metrics focus on their specific formulas while this class handles the plumbing.

## How It Works

External cluster metrics compare clustering results against known ground truth labels.
This base class provides common functionality shared by all external metrics including
contingency table construction, entropy calculations, and pair counting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExternalClusterMetricBase` | Initializes a new instance of the ExternalClusterMetricBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Binomial(Int32,Int32)` | Computes the binomial coefficient C(n, k). |
| `BuildContingencyTable(Vector<>,Vector<>)` | Builds a contingency table from true and predicted labels. |
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeEntropy(Vector<>)` | Computes the entropy of a label vector. |
| `ComputeEntropyFromCounts(Dictionary<Int32,Int32>,Int32)` | Computes the entropy of a label distribution. |
| `ComputeMutualInformation(Dictionary<ValueTuple<Int32,Int32>,Int32>,Dictionary<Int32,Int32>,Dictionary<Int32,Int32>,Int32)` | Computes mutual information from a contingency table. |
| `ComputePairConfusionMatrix(Vector<>,Vector<>)` | Computes the pair confusion matrix components. |
| `ValidateLabelVectors(Vector<>,Vector<>)` | Validates that the input label vectors have the same length. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Log2` | Precomputed log(2) for entropy calculations. |
| `NumOps` | The numeric operations instance for type T. |

