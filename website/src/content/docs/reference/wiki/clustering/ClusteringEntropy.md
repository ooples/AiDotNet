---
title: "ClusteringEntropy<T>"
description: "Entropy-based metrics for evaluating clustering against ground truth labels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Entropy-based metrics for evaluating clustering against ground truth labels.

## For Beginners

Entropy measures "How mixed are the clusters?"

Think of it like sorting socks:

- Low entropy: Each drawer has one color (easy to find socks!)
- High entropy: Each drawer is a random mix (chaos!)

For clustering:

- Entropy = 0: Perfect! Each cluster has only one class
- Entropy = log(C): Worst! Each cluster has equal parts of all classes

We want LOW entropy (pure clusters).

## How It Works

Entropy measures the "disorder" or "uncertainty" in cluster assignments.
Lower entropy indicates that clusters are more homogeneous with respect
to the true classes.

Entropy of cluster k = -sum over all classes j of p(j|k) * log(p(j|k))
Overall entropy = sum over all clusters k of (n_k/n) * Entropy(k)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusteringEntropy` | Initializes a new ClusteringEntropy instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeNormalized(Vector<>,Vector<>)` | Computes normalized entropy (0 = perfect, 1 = worst). |
| `ComputePerCluster(Vector<>,Vector<>)` | Computes entropy for each cluster. |

