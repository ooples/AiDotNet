---
title: "FedAdamServerOptimizer<T>"
description: "FedAdam server optimizer — adaptive learning rates with momentum and second-moment estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ServerOptimizers`

FedAdam server optimizer — adaptive learning rates with momentum and second-moment estimation.

## For Beginners

Adam combines the benefits of Adagrad (adaptive learning rates)
and momentum (smoothed gradient direction). FedAdam applies this at the server level,
treating aggregated client updates as pseudo-gradients for server-side optimization.

## How It Works

Reference: Reddi, S., et al. (2021). "Adaptive Federated Optimization." ICLR 2021.

