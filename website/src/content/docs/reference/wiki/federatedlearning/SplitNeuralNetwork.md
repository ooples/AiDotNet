---
title: "SplitNeuralNetwork<T>"
description: "Implements a split neural network for vertical federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Implements a split neural network for vertical federated learning.

## For Beginners

In VFL, the neural network is split into two parts:

## How It Works

The top model is a simple multi-layer perceptron (MLP) that takes the aggregated
embeddings as input and produces predictions.

**Aggregation modes:** Party embeddings can be combined via concatenation (default),
element-wise sum, attention weighting, or learned gating.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SplitNeuralNetwork(Int32,Int32,Int32,SplitModelOptions,Nullable<Int32>)` | Initializes a new instance of `SplitNeuralNetwork`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AggregationMode` |  |
| `NumberOfParties` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateEmbeddings(IReadOnlyList<Tensor<>>)` |  |
| `BackwardTopModel(Tensor<>,IReadOnlyList<Tensor<>>)` |  |
| `ForwardTopModel(Tensor<>)` |  |
| `GetTopModelParameters` |  |
| `SetTopModelParameters(IReadOnlyList<Tensor<>>)` |  |
| `UpdateFromGradient(Tensor<>,Double)` | Updates the top model parameters given the loss gradient. |
| `UpdateTopModelParameters(Double)` |  |

