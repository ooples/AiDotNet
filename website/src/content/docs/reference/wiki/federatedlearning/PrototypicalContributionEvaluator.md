---
title: "PrototypicalContributionEvaluator<T>"
description: "Prototypical contribution evaluator: measures client value using prototype representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Prototypical contribution evaluator: measures client value using prototype representations.

## For Beginners

Instead of expensive Shapley value computation (which requires
testing many model combinations), this method uses a shortcut: compare what each client
"knows" (represented as prototype vectors — like a summary of their data) against the
global model. Clients whose prototypes align well with the global direction but also bring
unique information score highest.

## How It Works

**How it works:**

**Advantage:** Constant cost per round O(N * D), where N = clients, D = model size.
Much cheaper than Shapley for large federations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypicalContributionEvaluator(ContributionEvaluationOptions)` | Initializes a new instance of `PrototypicalContributionEvaluator`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateContributions(Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |
| `IdentifyFreeRiders(Dictionary<Int32,Double>)` |  |

