---
title: "GradientAscentUnlearner<T>"
description: "Approximate unlearning via gradient ascent: reverses learning by ascending the loss on target data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Unlearning`

Approximate unlearning via gradient ascent: reverses learning by ascending the loss on target data.

## For Beginners

Normal training minimizes the loss (gradient descent — going downhill).
Gradient ascent does the opposite: it maximizes the loss on the target client's data, effectively
making the model "forget" what it learned from that client. Think of it as deliberately making the
model bad at the target client's data while keeping it good at everyone else's.

## How It Works

**How it works:**

**Speed:** Much faster than exact retraining (minutes vs. hours). Provides approximate
guarantees — the target client's influence is reduced but not provably zero.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientAscentUnlearner(FederatedUnlearningOptions)` | Initializes a new instance of `GradientAscentUnlearner`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Unlearn(Int32,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |

