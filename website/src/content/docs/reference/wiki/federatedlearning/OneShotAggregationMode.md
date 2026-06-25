---
title: "OneShotAggregationMode"
description: "Aggregation modes for One-Shot Federated Learning."
section: "API Reference"
---

`Enums` · `AiDotNet.FederatedLearning.Trainers`

Aggregation modes for One-Shot Federated Learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `EnsembleDistillation` | Ensemble distillation (average + diversity tracking). |
| `UniformAverage` | Uniform average (equal weight per client). |
| `WeightedAverage` | Weighted average by sample count (FedAvg-style). |

