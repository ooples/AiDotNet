---
title: "ShapleyValueEvaluator<T>"
description: "Exact Shapley value evaluator: computes each client's marginal contribution across all coalitions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Exact Shapley value evaluator: computes each client's marginal contribution across all coalitions.

## For Beginners

The Shapley value (from game theory, Nobel Prize 2012) is the fairest
way to divide credit among participants. It answers: "On average, how much does each client improve
the model when added to any possible subset of other clients?"

## How It Works

**How it works:**

**Warning:** Exact Shapley requires evaluating 2^N coalitions, making it impractical for
more than ~15 clients. Use `DataShapleyEvaluator` for larger federations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShapleyValueEvaluator(ContributionEvaluationOptions)` | Initializes a new instance of `ShapleyValueEvaluator`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateContributions(Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |
| `IdentifyFreeRiders(Dictionary<Int32,Double>)` |  |

