---
title: "DriftAdaptiveAggregator<T>"
description: "Drift-adaptive aggregator: wraps any aggregation strategy and adjusts weights based on drift."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.DriftDetection`

Drift-adaptive aggregator: wraps any aggregation strategy and adjusts weights based on drift.

## For Beginners

In standard FedAvg, all clients get weight proportional to their
data size. But if some clients are experiencing concept drift (their data has changed),
their updates may hurt the global model. This aggregator detects drifting clients and
reduces their influence while boosting stable clients.

## How It Works

**How it works:**

**Integration:** Use this as a wrapper around your existing aggregation strategy.
It modifies the weights before aggregation happens.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DriftAdaptiveAggregator(FederatedDriftOptions,IFederatedDriftDetector<>)` | Initializes a new instance of `DriftAdaptiveAggregator`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LatestReport` | Gets the latest drift report. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetClientsNeedingRetraining` | Gets clients that need selective retraining. |
| `GetDriftingClients` | Gets the set of currently drifting client IDs. |
| `ProcessRound(Int32,Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,Double>,Dictionary<Int32,Double>)` | Processes a round of FL by detecting drift and computing adaptive weights. |
| `Reset` | Resets all drift detection state. |

