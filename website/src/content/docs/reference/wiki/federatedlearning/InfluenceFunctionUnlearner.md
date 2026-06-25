---
title: "InfluenceFunctionUnlearner<T>"
description: "Influence function-based unlearning: mathematically estimates and subtracts a client's contribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Unlearning`

Influence function-based unlearning: mathematically estimates and subtracts a client's contribution.

## For Beginners

Influence functions answer the question: "How much would the model
change if we removed one client's data?" Instead of actually retraining, we mathematically
estimate the answer using the Hessian (second derivative) of the loss. It's like calculating
how much a building would shift if one support column were removed, without actually removing it.

## How It Works

**How it works:**

**Trade-off:** More accurate than gradient ascent for small removals (1-2 clients),
but the Hessian approximation degrades for large removals or highly non-convex models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InfluenceFunctionUnlearner(FederatedUnlearningOptions)` | Initializes a new instance of `InfluenceFunctionUnlearner`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Unlearn(Int32,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |

