---
title: "RetrievalModule<T>"
description: "Retrieval module for TabR (Retrieval-Augmented Tabular Learning)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Retrieval module for TabR (Retrieval-Augmented Tabular Learning).

## For Beginners

The retrieval module works like a smart lookup:

1. For each test sample, find similar training samples
2. Retrieve their features and labels
3. Use this information to help make better predictions

Think of it as "looking up similar past cases" before making a decision.

## How It Works

The retrieval module finds similar training examples to a query sample
and provides their features and labels as additional context for prediction.
This is similar to k-NN but integrated into a neural network.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RetrievalModule(Int32,Int32,Double)` | Initializes the retrieval module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasTrainingData` | Gets whether training data has been stored. |
| `NumNeighbors` | Gets the number of neighbors to retrieve. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearTrainingData` | Clears stored training data. |
| `Retrieve(Tensor<>)` | Retrieves similar samples for a batch of queries. |
| `StoreTrainingData(Tensor<>,Tensor<>,Tensor<>)` | Stores training data for retrieval. |

