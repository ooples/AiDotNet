---
title: "GOGGLEOptions<T>"
description: "Configuration options for GOGGLE (Generative mOdelling for tabular data by learninG reLational structurE), a graph-based VAE that learns feature dependency structure for generating realistic tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for GOGGLE (Generative mOdelling for tabular data by learninG
reLational structurE), a graph-based VAE that learns feature dependency structure
for generating realistic tabular data.

## For Beginners

GOGGLE discovers which features depend on each other and uses
this knowledge to generate better data:

1. Learns a "dependency map" — e.g., "Income depends on Education and Age"
2. Features that are connected share information through graph neural networks
3. This produces data where feature relationships are more realistic

Example:

## How It Works

GOGGLE combines a VAE with a graph neural network (GNN) to learn and exploit
feature dependencies:

- **Structure learning**: Learns a soft adjacency matrix representing feature dependencies
- **GNN encoder**: Uses message passing on the learned graph to encode features
- **VAE framework**: Generates data through a latent space with reparameterization
- **Structure loss**: Regularizes the learned graph for sparsity and DAG properties

Reference: "GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure"
(Liu et al., ICLR 2023)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `Epochs` | Gets or sets the number of training epochs. |
| `HiddenDimension` | Gets or sets the hidden dimension for GNN and MLP layers. |
| `KLWeight` | Gets or sets the weight for the KL divergence loss term. |
| `LatentDimension` | Gets or sets the dimension of the VAE latent space. |
| `LearningRate` | Gets or sets the learning rate. |
| `NumGNNLayers` | Gets or sets the number of GNN message-passing layers. |
| `SparsityWeight` | Gets or sets the sparsity regularization for the adjacency matrix. |
| `StructureWeight` | Gets or sets the weight for the graph structure learning loss. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column transformation. |

