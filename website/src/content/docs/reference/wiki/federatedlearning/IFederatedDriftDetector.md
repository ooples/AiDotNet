---
title: "IFederatedDriftDetector<T>"
description: "Detects concept drift in federated learning by monitoring client model updates over time."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.DriftDetection`

Detects concept drift in federated learning by monitoring client model updates over time.

## For Beginners

In production FL, data doesn't stay static. Customer behavior changes,
new fraud patterns emerge, market conditions shift. This interface monitors each client's training
behavior and flags when their data distribution has changed significantly. The system can then
adapt (e.g., increase that client's training epochs, adjust aggregation weights, or trigger
selective retraining).

## How It Works

**Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` | Gets the name of the drift detection method. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectDrift(Int32,Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,Double>)` | Detects drift across all federated clients for the current round. |
| `GetAdaptiveWeights(Dictionary<Int32,Double>,DriftReport)` | Gets recommended aggregation weight adjustments based on detected drift. |
| `Reset` | Resets drift detection state for all clients. |

