---
title: "PersonalizedLayerSelectionStrategy"
description: "Strategy for selecting which layers to personalize in federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.FederatedLearning.Personalization`

Strategy for selecting which layers to personalize in federated learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `ByPattern` | Personalize layers matching user-provided name patterns (e.g., "batch_norm", "classifier"). |
| `LastN` | Personalize the last N% of layers (sorted by ordinal name). |

