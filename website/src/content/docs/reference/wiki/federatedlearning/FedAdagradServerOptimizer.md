---
title: "FedAdagradServerOptimizer<T>"
description: "FedAdagrad server optimizer — adaptive learning rates using accumulated squared gradients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ServerOptimizers`

FedAdagrad server optimizer — adaptive learning rates using accumulated squared gradients.

## For Beginners

Adagrad automatically reduces the learning rate for parameters
that have been updated frequently, allowing rarely-updated parameters to learn faster.
In federated settings, this helps handle heterogeneous client updates.

## How It Works

Reference: Reddi, S., et al. (2021). "Adaptive Federated Optimization." ICLR 2021.

