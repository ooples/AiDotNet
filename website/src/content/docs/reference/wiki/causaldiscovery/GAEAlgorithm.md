---
title: "GAEAlgorithm<T>"
description: "GAE — Graph Autoencoder for causal discovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

GAE — Graph Autoencoder for causal discovery.

## For Beginners

A Graph Autoencoder compresses data through a "graph bottleneck."
The connections in this bottleneck represent causal relationships — the autoencoder
is forced to find the minimal set of connections needed to recreate the data.

## How It Works

GAE uses an autoencoder architecture where the encoder produces a latent graph
representation and the decoder reconstructs the data through the learned graph.
The encoder maps each variable to separate source/target latent embeddings via shared MLP,
then computes edge probabilities as sigmoid(Zs_i^T * Zt_j) using asymmetric dot products
to encode directionality. The decoder reconstructs X_hat = X * A where A is the soft
adjacency. NOTEARS acyclicity constraint h(A) = tr(exp(A∘A)) - d ensures a valid DAG.

Reference: Kipf and Welling (2016), "Variational Graph Auto-Encoders", NeurIPS Workshop.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `SampleStandardNormal(Random)` | Samples from a standard normal distribution N(0,1) using Box-Muller transform. |

