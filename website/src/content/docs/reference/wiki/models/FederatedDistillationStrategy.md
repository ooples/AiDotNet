---
title: "FederatedDistillationStrategy"
description: "Specifies the federated knowledge distillation strategy."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the federated knowledge distillation strategy.

## Fields

| Field | Summary |
|:-----|:--------|
| `FedDF` | FedDF — ensemble distillation using unlabeled public data. |
| `FedGEN` | FedGEN — data-free distillation using server-side generative model. |
| `FedMD` | FedMD — mutual distillation on a public dataset. |
| `None` | No distillation — standard parameter aggregation. |

