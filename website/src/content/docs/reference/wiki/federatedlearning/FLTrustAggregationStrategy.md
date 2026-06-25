---
title: "FLTrustAggregationStrategy<T>"
description: "Implements the FLTrust aggregation strategy for Byzantine-robust federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements the FLTrust aggregation strategy for Byzantine-robust federated learning.

## For Beginners

In standard federated learning, malicious clients can send
fake updates to corrupt the global model. FLTrust solves this by having the server maintain
a small, clean "root" dataset. The server computes its own gradient on this data, then
scores each client's update by how similar its direction is to the server's gradient.
Only client updates that point in roughly the same direction as the server's are included,
and they are re-scaled to the server gradient's magnitude to prevent magnitude attacks.

## How It Works

Trust score computation:

Reference: Cao, X., et al. (2021). "FLTrust: Byzantine-robust Federated Learning
via Trust Bootstrapping." NDSS 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FLTrustAggregationStrategy` | Initializes a new instance of the `FLTrustAggregationStrategy` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `GetStrategyName` |  |
| `SetServerGradient(Dictionary<String,[]>)` | Sets the server's root-dataset gradient used as the trust anchor. |

