---
title: "DiffusiveNoiseUnlearner<T>"
description: "Unlearning via structured diffusive noise injection targeting memorized samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Unlearning`

Unlearning via structured diffusive noise injection targeting memorized samples.

## For Beginners

This approach (from 2025 research) works like adding carefully designed
static to a radio signal to drown out a specific station. Instead of reversing the learning directly,
we inject noise that is specifically structured to disrupt the model's memorization of the target
client's data patterns. Then we "heal" the model by fine-tuning on remaining clients.

## How It Works

**How it works:**

**Advantages:** More robust than gradient ascent (which can overshoot), and doesn't require
Hessian computation (which is expensive for large models). The structured noise targets memorization
specifically rather than general model quality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusiveNoiseUnlearner(FederatedUnlearningOptions)` | Initializes a new instance of `DiffusiveNoiseUnlearner`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Unlearn(Int32,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |

