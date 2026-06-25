---
title: "IClientContributionEvaluator<T>"
description: "Evaluates how much each client contributed to the federated global model."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Fairness`

Evaluates how much each client contributed to the federated global model.

## For Beginners

In a group project, some students do more work than others.
This interface measures each client's "fair share" of the global model improvement.
High-contribution clients provided valuable data, while low-contribution clients may be
free-riding (benefiting without contributing). This is essential for fair compensation
and for detecting poisoned or low-quality updates.

## How It Works

**Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` | Gets the name of this contribution evaluation method. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateContributions(Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` | Evaluates the contribution of each client to the global model. |
| `IdentifyFreeRiders(Dictionary<Int32,Double>)` | Identifies free-rider clients whose contribution falls below the threshold. |

