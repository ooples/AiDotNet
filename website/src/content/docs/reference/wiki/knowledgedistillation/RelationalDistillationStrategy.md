---
title: "RelationalDistillationStrategy<T>"
description: "RelationalDistillationStrategy<T> — Models & Types in AiDotNet.KnowledgeDistillation.Strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RelationalDistillationStrategy(Double,Double,Double,Double,Int32,RelationalDistanceMetric)` | Initializes a new instance of the RelationalDistillationStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAnalyticalAngleGradient(Vector<>,Vector<>,Vector<>)` | Computes the analytical gradient of angle θ at vertex j with respect to vector i. |
| `ComputeAngle(Vector<>,Vector<>,Vector<>)` | Computes angle between three points (angle at point j). |
| `ComputeAngleWiseLoss(Vector<>[],Vector<>[],Int32)` | Computes angle-wise relational loss (preserves angular relationships). |
| `ComputeAverageRelationalGradientForBatch(Vector<>[],Vector<>[])` | Computes the average relational gradient for all samples in a batch. |
| `ComputeDistance(Vector<>,Vector<>)` | Computes distance between two vectors based on the selected metric. |
| `ComputeDistanceWiseLoss(Vector<>[],Vector<>[],Int32)` | Computes distance-wise relational loss (preserves pairwise distances). |
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes gradient of combined output loss and relational loss. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes combined output loss and relational loss. |
| `ComputePairwiseDistanceGradient(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes gradient of distance-wise loss for a pair. |
| `ComputePerSampleRelationalGradients(Vector<>[],Vector<>[])` | Computes per-sample relational gradients using finite differences on the relational loss. |
| `ComputeRelationalGradient(Vector<>,Vector<>,List<Vector<>>,List<Vector<>>)` | Computes gradient of relational loss with respect to a single student output. |
| `ComputeRelationalLoss(Vector<>[],Vector<>[])` | Computes relational knowledge distillation loss for a batch of samples. |
| `ComputeTripletAngleGradient(Vector<>,Vector<>,Vector<>,Vector<>,Vector<>,Vector<>)` | Computes gradient of angle-wise loss for a triplet using analytical gradient. |
| `Reset` | Resets the strategy's internal state. |

