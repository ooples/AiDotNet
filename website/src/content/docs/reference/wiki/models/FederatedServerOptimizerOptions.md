---
title: "FederatedServerOptimizerOptions"
description: "Configuration options for server-side federated optimizers (FedOpt family)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for server-side federated optimizers (FedOpt family).

## How It Works

**For Beginners:** These optimizers run on the server after aggregation to update the global model.
If `Optimizer` is `None`, the server uses the aggregated parameters directly (FedAvg-style).

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta1` | Gets or sets the beta1 coefficient (Adam/Yogi). |
| `Beta2` | Gets or sets the beta2 coefficient (Adam/Yogi). |
| `Epsilon` | Gets or sets the epsilon value used for numerical stability. |
| `LearningRate` | Gets or sets the server learning rate (step size). |
| `Momentum` | Gets or sets the server momentum coefficient for FedAvgM. |
| `Optimizer` | Gets or sets the server optimizer. |

