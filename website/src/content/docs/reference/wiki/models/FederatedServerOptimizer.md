---
title: "FederatedServerOptimizer"
description: "Specifies which server-side federated optimizer (FedOpt family) to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies which server-side federated optimizer (FedOpt family) to use.

## How It Works

**For Beginners:** FedOpt optimizers run on the server after aggregation and can improve convergence.

## Fields

| Field | Summary |
|:-----|:--------|
| `FedAdagrad` | FedAdagrad server optimizer. |
| `FedAdam` | FedAdam server optimizer. |
| `FedAvgM` | FedAvg with server momentum. |
| `FedYogi` | FedYogi server optimizer. |
| `None` | No server optimizer (use aggregated parameters directly). |

