---
title: "EdgeOptimizer<T, TInput, TOutput>"
description: "Optimizer for edge device deployment with ARM NEON and other optimizations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Edge`

Optimizer for edge device deployment with ARM NEON and other optimizations.
Properly integrates with IFullModel architecture.

## For Beginners

EdgeOptimizer provides AI safety functionality. Default values follow the original paper settings.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateLoadBalancedPartitionPoint(ILayeredModel<>)` | Calculates the optimal partition point by balancing estimated FLOPs between edge and cloud. |
| `CalculateProportionalPartitionPoint(ILayeredModel<>,Double)` | Calculates a partition point at a fixed proportion of the model's layers. |
| `CreateAdaptiveConfig(Double,Double)` | Applies adaptive inference optimization (quality vs speed tradeoff). |
| `OptimizeForEdge(IFullModel<,,>)` | Optimizes a model for edge deployment. |
| `PartitionModel(IFullModel<,,>)` | Partitions a model for split execution between cloud and edge. |

