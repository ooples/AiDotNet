---
title: "IGraphConvolutionLayer<T>"
description: "Defines the contract for graph convolutional layers that process graph-structured data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for graph convolutional layers that process graph-structured data.

## For Beginners

This interface defines what all graph layers must be able to do.

Graph layers are special because they work with data that has connections:

- Social networks (people connected to friends)
- Molecules (atoms connected by bonds)
- Transportation networks (cities connected by roads)
- Knowledge graphs (concepts connected by relationships)

The key difference from regular layers is that graph layers need to know
which nodes are connected to which other nodes. That's what the adjacency matrix provides.

## How It Works

Graph convolutional layers process data that is organized as graphs (nodes and edges).
This interface extends the base layer interface with graph-specific functionality,
particularly the ability to work with adjacency matrices that define graph structure.

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` | Gets the number of input features per node. |
| `OutputFeatures` | Gets the number of output features per node. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAdjacencyMatrix` | Gets the adjacency matrix currently being used by this layer. |
| `SetAdjacencyMatrix(Tensor<>)` | Sets the adjacency matrix that defines the graph structure. |

