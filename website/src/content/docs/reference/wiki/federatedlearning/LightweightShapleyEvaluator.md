---
title: "LightweightShapleyEvaluator<T>"
description: "Implements Lightweight Shapley — O(n) Shapley value approximation using gradient similarity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Implements Lightweight Shapley — O(n) Shapley value approximation using gradient similarity.

## For Beginners

Exact Shapley values require evaluating all 2^n subsets of
clients — impossibly expensive for even 20 clients. Lightweight Shapley approximates
each client's contribution by measuring how similar their gradient is to the ideal global
gradient. Clients whose updates are well-aligned with the consensus get high Shapley values,
while adversarial or low-quality updates get low values. This runs in O(n) time.

## How It Works

Approximation:

Reference: Lightweight Shapley for Federated Contribution Evaluation (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LightweightShapleyEvaluator(Double)` | Creates a new Lightweight Shapley evaluator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FreeRiderThreshold` | Gets the free-rider threshold. |
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateContributions(Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |
| `IdentifyFreeRiders(Dictionary<Int32,Double>)` |  |

