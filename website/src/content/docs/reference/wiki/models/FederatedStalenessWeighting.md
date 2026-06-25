---
title: "FederatedStalenessWeighting"
description: "Specifies how to down-weight stale updates in asynchronous federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how to down-weight stale updates in asynchronous federated learning.

## How It Works

**For Beginners:** If an update was computed on an old global model, it can be less helpful.
Staleness weighting reduces the influence of older updates.

## Fields

| Field | Summary |
|:-----|:--------|
| `Constant` | No additional staleness weighting (constant weight). |
| `Exponential` | Weight = exp(-rate * staleness). |
| `Inverse` | Weight = 1 / (1 + staleness). |
| `Polynomial` | Weight = 1 / (1 + staleness)^rate. |

