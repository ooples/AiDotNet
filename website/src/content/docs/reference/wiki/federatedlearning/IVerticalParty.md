---
title: "IVerticalParty<T>"
description: "Represents a party in vertical federated learning that holds a subset of features."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Vertical`

Represents a party in vertical federated learning that holds a subset of features.

## For Beginners

In vertical FL, each party holds different features (columns)
for the same entities (rows). For example, a bank has income and credit score columns,
while a hospital has diagnosis and prescription columns. Each party runs a local "bottom model"
on its features to produce an embedding (a compressed representation).

## How It Works

The party interface abstracts away the local computation: the VFL trainer asks each party
to compute forward passes on its local data and backward passes using gradients received
from the coordinator.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the output dimension of this party's bottom model (embedding size). |
| `FeatureCount` | Gets the number of features held by this party. |
| `IsLabelHolder` | Gets whether this party holds the labels for supervised learning. |
| `PartyId` | Gets the unique identifier for this party. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBackward(Tensor<>,Double)` | Applies the backward pass using gradients received from the coordinator. |
| `ComputeForward(IReadOnlyList<Int32>)` | Computes the forward pass of this party's bottom model on the given entity indices. |
| `GetEntityIds` | Gets the entity IDs that this party has data for. |
| `GetParameters` | Gets the current parameters of this party's bottom model for checkpointing. |
| `SetParameters(IReadOnlyList<Tensor<>>)` | Sets the parameters of this party's bottom model (for loading checkpoints or unlearning). |

