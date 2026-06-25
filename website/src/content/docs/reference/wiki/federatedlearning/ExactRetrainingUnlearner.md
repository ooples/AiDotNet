---
title: "ExactRetrainingUnlearner<T>"
description: "Gold-standard unlearning: retrains the model from scratch excluding the target client."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Unlearning`

Gold-standard unlearning: retrains the model from scratch excluding the target client.

## For Beginners

The most reliable way to forget a client is to pretend they never
existed and retrain everything from the beginning. This is like erasing someone's name from every
page of a book and rewriting the book — it's perfect but very slow.

## How It Works

**How it works:**

**When to use:** Regulatory audits requiring provable unlearning, or when approximate
methods fail verification. Very expensive — O(R * C) where R = rounds, C = clients.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExactRetrainingUnlearner(FederatedUnlearningOptions)` | Initializes a new instance of `ExactRetrainingUnlearner`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Unlearn(Int32,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` |  |

