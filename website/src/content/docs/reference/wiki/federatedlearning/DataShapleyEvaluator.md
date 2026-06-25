---
title: "DataShapleyEvaluator<T>"
description: "Data Shapley evaluator: efficient Monte Carlo approximation of Shapley values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Data Shapley evaluator: efficient Monte Carlo approximation of Shapley values.

## For Beginners

Exact Shapley values require evaluating all possible subsets (2^N),
which is impractical for more than ~15 clients. Data Shapley instead randomly samples client
orderings (permutations) and averages the marginal contributions. With enough samples, it
converges to the true Shapley value while running in polynomial time.

## How It Works

**How it works:**

**Recommended for:** Federations with 10+ clients where exact Shapley is too expensive.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataShapleyEvaluator(ContributionEvaluationOptions)` | Initializes a new instance of `DataShapleyEvaluator`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateContributions(Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |
| `IdentifyFreeRiders(Dictionary<Int32,Double>)` |  |

