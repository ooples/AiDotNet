---
title: "FederatedContinualLearningStrategy"
description: "Specifies the federated continual learning strategy for preventing catastrophic forgetting across rounds."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the federated continual learning strategy for preventing catastrophic forgetting across rounds.

## Fields

| Field | Summary |
|:-----|:--------|
| `DataFreeFCL` | Data-Free FCL — prevents forgetting using synthetic data from teacher model, no real data storage. |
| `ExperienceReplay` | Experience Replay — reservoir sampling buffer of representative old samples per client. |
| `FedAGC` | FedAGC — Adaptive Gradient Correction balancing plasticity and stability. |
| `FedCIL` | FedCIL — class-incremental learning with prototype-based knowledge consolidation. |
| `FederatedEWC` | Federated EWC — Fisher-information-based importance weighting aggregated across clients. |
| `None` | No federated continual learning — standard aggregation without forgetting prevention. |
| `OrthogonalProjection` | Federated Orthogonal Projection — projects gradients orthogonal to important directions. |

