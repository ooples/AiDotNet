---
title: "VerticalPartyClient<T>"
description: "Represents a feature-holding party in vertical federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Represents a feature-holding party in vertical federated learning.

## For Beginners

This is one of the parties in a VFL setup that holds some features
(columns) of the dataset but NOT the labels (prediction targets). For example, a bank that
has income and credit score data but doesn't know patient outcomes.

## How It Works

The party runs a local "bottom model" (a small neural network) on its features to produce
an embedding (compressed representation) that is sent to the coordinator. During backpropagation,
the party receives gradients for its embedding and uses them to update its local model.

**Privacy guarantee:** The raw features never leave the party. Only the embedding
(and its gradients) cross party boundaries. The embedding is a lossy compression that
makes it difficult to reconstruct the original features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VerticalPartyClient(String,Tensor<>,IReadOnlyList<String>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of `VerticalPartyClient`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `FeatureCount` |  |
| `IsLabelHolder` |  |
| `PartyId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBackward(Tensor<>,Double)` |  |
| `ComputeForward(IReadOnlyList<Int32>)` |  |
| `GetEntityIds` |  |
| `GetParameters` |  |
| `SetParameters(IReadOnlyList<Tensor<>>)` |  |

