---
title: "GroupFairnessConstraint<T>"
description: "Enforces group fairness constraints during federated learning aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Enforces group fairness constraints during federated learning aggregation.

## For Beginners

Imagine a federation of hospitals: 8 large urban hospitals and 2 small
rural clinics. Without fairness constraints, the global model optimizes mostly for urban patients
(the majority). This constraint ensures rural patients get comparable model quality by adjusting
how much weight each group receives during aggregation.

## How It Works

**Supported constraints:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroupFairnessConstraint(FederatedFairnessOptions)` | Initializes a new instance of `GroupFairnessConstraint`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConstraintName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustWeights(Dictionary<Int32,Double>,Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,Int32>)` |  |
| `EvaluateFairness(Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,Int32>)` |  |
| `IsSatisfied(Double)` |  |

