---
title: "Purity<T>"
description: "Purity metric for evaluating clustering against ground truth labels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Purity metric for evaluating clustering against ground truth labels.

## For Beginners

Purity asks "How pure are the clusters?"

For each cluster:

- Find the most common true class
- Count how many points belong to that class

Purity = Total correctly assigned / Total points

Example:

- Cluster 1: 30 cats, 10 dogs → 30 correct (majority is cats)
- Cluster 2: 20 cats, 40 dogs → 40 correct (majority is dogs)
- Purity = (30 + 40) / 100 = 0.70

Higher purity = Better clustering!
But beware: Purity increases as k increases (k=n gives purity=1).

## How It Works

Purity measures the fraction of correctly assigned points, where "correct"
means each cluster is assigned to its majority class. Values range from
1/k (random) to 1 (perfect).

Purity = (1/n) * sum over all clusters k of max_j |c_k ∩ t_j|
Where c_k is cluster k and t_j is true class j.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Purity` | Initializes a new Purity instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputePerCluster(Vector<>,Vector<>)` | Computes purity per cluster. |

