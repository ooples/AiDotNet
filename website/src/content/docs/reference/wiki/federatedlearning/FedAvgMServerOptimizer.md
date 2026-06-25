---
title: "FedAvgMServerOptimizer<T>"
description: "FedAvgM server optimizer — server-side momentum for stabilized federated averaging."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ServerOptimizers`

FedAvgM server optimizer — server-side momentum for stabilized federated averaging.

## For Beginners

Momentum helps smooth updates across rounds. Instead of applying only the
current round's update, the server maintains a running "velocity" that accumulates updates.
This reduces oscillations caused by heterogeneous client data.

## How It Works

Reference: Hsu, T.-M. H., et al. (2019). "Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification." NeurIPS Workshop 2019.

