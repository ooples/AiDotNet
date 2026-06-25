---
title: "FederatedDistillationOptions"
description: "Configuration options for federated knowledge distillation, enabling model-heterogeneous FL."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated knowledge distillation, enabling model-heterogeneous FL.

## For Beginners

In standard federated learning, all clients must use the exact same model
architecture. Knowledge distillation removes this constraint — a phone can use a tiny model while a
server uses a large model, and they still learn from each other.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationAlpha` | Gets or sets the weight for distillation loss vs task-specific loss. |
| `DistillationEpochs` | Gets or sets the number of local distillation epochs per round. |
| `FeatureDimension` | Gets or sets the feature dimensionality for FedGEN synthetic data. |
| `NumClasses` | Gets or sets the number of output classes for FedGEN. |
| `Strategy` | Gets or sets the distillation strategy. |
| `Temperature` | Gets or sets the softmax temperature for soft label generation. |

