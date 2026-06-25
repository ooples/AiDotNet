---
title: "FederatedHeterogeneityCorrection"
description: "Specifies which heterogeneity correction algorithm to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies which heterogeneity correction algorithm to use.

## How It Works

**For Beginners:** Client data can be non-IID and clients may perform different amounts of local work.
Heterogeneity correction methods reduce client drift and improve convergence.

## Fields

| Field | Summary |
|:-----|:--------|
| `FedDyn` | FedDyn dynamic regularization. |
| `FedNova` | FedNova normalization. |
| `None` | No heterogeneity correction. |
| `Scaffold` | SCAFFOLD control variates. |

