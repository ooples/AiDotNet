---
title: "FScore<T>"
description: "F-Score metric for 3D reconstruction evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

F-Score metric for 3D reconstruction evaluation.

## How It Works

F-Score combines precision and recall at a given distance threshold.
Precision = fraction of predicted points within threshold of a ground truth point.
Recall = fraction of ground truth points within threshold of a predicted point.
F-Score = 2 * (Precision * Recall) / (Precision + Recall)

**Usage in 3D AI:**

- 3D reconstruction quality evaluation
- Mesh surface accuracy assessment
- Point cloud completion evaluation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FScore(Double)` | Initializes a new instance of the F-Score metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes F-Score between predicted and ground truth point clouds. |
| `ComputePrecisionRecall(Tensor<>,Tensor<>)` | Computes precision and recall separately. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider for type T. |
| `_threshold` | Distance threshold for considering a point as correctly reconstructed. |

