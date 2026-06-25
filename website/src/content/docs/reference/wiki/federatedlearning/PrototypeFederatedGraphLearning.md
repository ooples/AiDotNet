---
title: "PrototypeFederatedGraphLearning<T>"
description: "Prototype-based federated graph learning: clients share class prototypes instead of full model parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

Prototype-based federated graph learning: clients share class prototypes instead of full model parameters.

## For Beginners

Sharing full GNN model parameters raises two concerns: (1) model
parameters can leak information about the training graph, and (2) different clients may have
very different graph structures that benefit from different model architectures. Prototype-based
FGL addresses both:

## How It Works

**Benefits:** Prototypes are much smaller than full models (K prototypes of dimension D vs.
millions of parameters), more robust to topology heterogeneity, and reveal less about graph structure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypeFederatedGraphLearning(FederatedGraphOptions,Int32,Int32)` | Initializes a new instance of `PrototypeFederatedGraphLearning`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GlobalPrototypes` | Gets the current global prototypes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregatePrototypes` | Aggregates prototypes from all registered clients into global prototypes. |
| `ComputePrototypeLoss(Dictionary<Int32,Tensor<>>,Double)` | Computes prototype regularization loss: pull client prototypes toward global prototypes. |
| `ComputePrototypes(Tensor<>,Tensor<>,Int32)` | Computes a client's class prototypes from node embeddings and labels. |
| `RegisterClientPrototypes(Int32,Dictionary<Int32,Tensor<>>)` | Registers a client's class prototypes for the current round. |

