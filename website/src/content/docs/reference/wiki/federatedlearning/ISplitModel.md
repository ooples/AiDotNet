---
title: "ISplitModel<T>"
description: "Represents a split neural network for vertical federated learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Vertical`

Represents a split neural network for vertical federated learning.

## For Beginners

A split neural network is divided into two parts:

## How It Works

The "split point" is where the network is divided. Below the split, computation is
local and private. Above the split, computation uses combined information from all parties.

During training, forward passes flow upward (bottom -> top) and gradients flow
downward (top -> bottom). Only the embeddings and their gradients cross party boundaries,
never the raw features.

## Properties

| Property | Summary |
|:-----|:--------|
| `AggregationMode` | Gets the aggregation mode used to combine party embeddings. |
| `NumberOfParties` | Gets the number of parties this split model is configured for. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateEmbeddings(IReadOnlyList<Tensor<>>)` | Aggregates embeddings from multiple parties according to the configured aggregation mode. |
| `BackwardTopModel(Tensor<>,IReadOnlyList<Tensor<>>)` | Computes the top model backward pass and returns gradients for each party's embedding. |
| `ForwardTopModel(Tensor<>)` | Computes the top model forward pass on the combined embeddings. |
| `GetTopModelParameters` | Gets the current parameters of the top model for checkpointing. |
| `SetTopModelParameters(IReadOnlyList<Tensor<>>)` | Sets the top model parameters (for loading checkpoints or unlearning). |
| `UpdateTopModelParameters(Double)` | Updates the top model parameters using the computed gradients. |

