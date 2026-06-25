---
title: "IFairnessConstraint<T>"
description: "Defines and enforces fairness constraints during federated learning aggregation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Fairness`

Defines and enforces fairness constraints during federated learning aggregation.

## For Beginners

A fairness constraint ensures the global model works well for ALL
client groups, not just the majority. Without it, a model trained across 10 urban hospitals
and 2 rural hospitals might be great for urban patients but poor for rural ones. Fairness
constraints rebalance the aggregation to protect underrepresented groups.

## How It Works

**Common fairness measures:**

## Properties

| Property | Summary |
|:-----|:--------|
| `ConstraintName` | Gets the name of this fairness constraint. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustWeights(Dictionary<Int32,Double>,Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,Int32>)` | Adjusts aggregation weights to enforce fairness constraints. |
| `EvaluateFairness(Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,Int32>)` | Evaluates the current fairness metric across client groups. |
| `IsSatisfied(Double)` | Gets whether the current model satisfies the fairness constraint. |

