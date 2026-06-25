---
title: "FederatedFSSelector<T>"
description: "Federated Feature Selection Selector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Privacy`

Federated Feature Selection Selector.

## For Beginners

Federated learning keeps data on local devices/servers
(clients) and only shares model updates. This selector simulates that by:
1) Splitting data into partitions (simulating different clients)
2) Computing feature importance locally on each partition
3) Aggregating scores across clients without sharing raw data
This provides privacy as raw data never leaves its partition.

## How It Works

Simulates federated feature selection where data is partitioned across
multiple clients, and feature importance is computed locally then aggregated.

