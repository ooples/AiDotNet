---
title: "DAGGNNAlgorithm<T>"
description: "DAG-GNN — DAG Structure Learning with Graph Neural Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

DAG-GNN — DAG Structure Learning with Graph Neural Networks.

## For Beginners

DAG-GNN trains a special neural network (GNN) to simultaneously
figure out the graph structure AND generate data that matches the observed data.
The best graph is the one that lets the network most accurately recreate the data.

## How It Works

DAG-GNN uses a variational autoencoder framework where the encoder maps data to a
latent adjacency matrix A via learned node embeddings, and the decoder reconstructs
data using X_hat = X * A. The NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d
is enforced via augmented Lagrangian. Edge probabilities are computed as
A[i,j] = sigmoid(Zs_i^T * Zt_j) from separate source/target embeddings Zs, Zt.

Reference: Yu et al. (2019), "DAG-GNN: DAG Structure Learning with Graph Neural Networks", ICML.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

