---
title: "IFederatedUnlearner<T>"
description: "Core interface for federated unlearning: removes a client's contribution from the global model."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Unlearning`

Core interface for federated unlearning: removes a client's contribution from the global model.

## For Beginners

This interface defines the "forget" operation for federated learning.
When a client exercises their GDPR right to be forgotten, you call `Tensor{` with
their client ID. The unlearner modifies the global model to remove that client's influence
and returns a certificate proving it was done.

## How It Works

**Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` | Gets the name of this unlearning method. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Unlearn(Int32,Tensor<>,Dictionary<Int32,List<Tensor<>>>)` | Removes a client's contribution from the global model. |

