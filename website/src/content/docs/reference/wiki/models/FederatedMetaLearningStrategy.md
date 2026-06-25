---
title: "FederatedMetaLearningStrategy"
description: "Specifies the federated meta-learning strategy."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the federated meta-learning strategy.

## For Beginners

Each strategy determines how the server uses client adaptation results
to update the global initialization for better per-client fine-tuning.

## Fields

| Field | Summary |
|:-----|:--------|
| `FedMAML` | FedMAML — first-order MAML approximation; full second-order requires explicit gradient support. |
| `None` | No meta-learning — standard aggregation of client updates. |
| `PerFedAvg` | PerFedAvg — per-client FedAvg treated as a Reptile-style first-order update. |
| `Reptile` | Reptile — first-order meta-update based on post-adaptation parameters. |

