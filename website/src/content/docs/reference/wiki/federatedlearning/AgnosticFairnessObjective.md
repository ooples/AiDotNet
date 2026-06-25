---
title: "AgnosticFairnessObjective<T>"
description: "Implements AFL (Agnostic Federated Learning) — minimax fairness optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Implements AFL (Agnostic Federated Learning) — minimax fairness optimization.

## For Beginners

Standard FL minimizes the average loss across all clients.
This can leave some clients with terrible performance (e.g., a hospital with rare diseases).
AFL instead optimizes for the worst-performing client — it's "agnostic" to which client
distribution the model will be tested on. This ensures no client is left behind, at the
cost of slightly lower average performance.

## How It Works

Objective:

The lambda weights dynamically increase for clients with high loss.

Reference: Mohri, M., et al. (2019). "Agnostic Federated Learning." ICML 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgnosticFairnessObjective(Double)` | Creates a new AFL objective. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LambdaLearningRate` | Gets the lambda learning rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeWeights(Dictionary<Int32,Double>)` | Computes AFL aggregation weights based on client losses. |

