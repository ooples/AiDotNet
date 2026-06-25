---
title: "FMeasure<T>"
description: "F-Measure (F-Score) for comparing clustering results against ground truth."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

F-Measure (F-Score) for comparing clustering results against ground truth.

## For Beginners

F-Measure asks "How well do clusters match classes?"

For each true class:

- Find which cluster matches it best
- Calculate how well that cluster captures the class

Precision: "Of the cluster, how many belong to this class?"
Recall: "Of this class, how many are in this cluster?"
F-Measure: Balances both (harmonic mean)

Higher is better! 1.0 = perfect, 0 = no agreement.

## How It Works

The F-Measure combines precision and recall for clustering evaluation.
For each true class, it finds the best matching cluster, then averages
the F-scores weighted by class sizes.

F(C_i, K_j) = (2 * Precision * Recall) / (Precision + Recall)
Where:

- Precision = |C_i ∩ K_j| / |K_j|
- Recall = |C_i ∩ K_j| / |C_i|

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FMeasure(Double)` | Initializes a new FMeasure instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeBCubed(Vector<>,Vector<>)` | Computes pair-counting based F-Measure (BCubed F-Measure). |
| `ComputeMatrix(Vector<>,Vector<>)` | Computes the F-Measure matrix between all classes and clusters. |

