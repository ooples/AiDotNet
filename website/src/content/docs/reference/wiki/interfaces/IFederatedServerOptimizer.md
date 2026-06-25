---
title: "IFederatedServerOptimizer<T>"
description: "Applies a server-side optimization step in federated learning (FedOpt family)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Applies a server-side optimization step in federated learning (FedOpt family).

## How It Works

**For Beginners:** In classic FedAvg, the server simply replaces the global model with the averaged client model.
In FedOpt, the server treats the aggregated update like a "gradient" and applies an optimizer step (like Adam),
which can improve stability and convergence, especially with non-IID data.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptimizerName` | Gets the name of the server optimizer. |
| `Step(Vector<>,Vector<>)` | Updates global parameters given the current parameters and an aggregated target. |

