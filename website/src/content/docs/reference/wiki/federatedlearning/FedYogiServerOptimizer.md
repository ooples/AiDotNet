---
title: "FedYogiServerOptimizer<T>"
description: "FedYogi server optimizer — adaptive learning rates with controlled second-moment growth."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ServerOptimizers`

FedYogi server optimizer — adaptive learning rates with controlled second-moment growth.

## For Beginners

Yogi is similar to Adam but more stable with noisy or sparse updates.
While Adam's second moment can grow unboundedly (causing vanishingly small learning rates),
Yogi uses an additive update that only grows the second moment when the gradient magnitude
exceeds the current estimate, preventing premature convergence.

## How It Works

Reference: Reddi, S., et al. (2021). "Adaptive Federated Optimization." ICLR 2021.

