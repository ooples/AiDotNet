---
title: "GraphWaveNetOptions<T>"
description: "Configuration options for GraphWaveNet (Graph WaveNet for Deep Spatial-Temporal Modeling)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for GraphWaveNet (Graph WaveNet for Deep Spatial-Temporal Modeling).

## For Beginners

GraphWaveNet achieves state-of-the-art traffic forecasting by combining:

**The Key Insight:**
Traffic patterns have two types of dependencies: spatial (how one road affects nearby roads)
and temporal (how patterns evolve over time). GraphWaveNet uses diffusion convolution
for spatial modeling and dilated causal convolutions for temporal modeling.

**What Problems Does GraphWaveNet Solve?**

- Traffic speed/flow prediction on road networks
- Air quality forecasting across sensor networks
- Electricity load prediction across power grids
- Any spatio-temporal forecasting with graph structure

**How GraphWaveNet Works:**

1. **Adaptive Adjacency:** Learns graph structure from node embeddings
2. **Diffusion Convolution:** Bidirectional message passing on the graph
3. **Dilated Temporal Conv:** WaveNet-style gated convolutions with exponentially growing dilation
4. **Skip Connections:** Collects outputs from all layers for final prediction

**GraphWaveNet Architecture:**

- Node Embeddings: Learnable E1, E2 for adaptive graph A = softmax(ReLU(E1*E2^T))
- Diffusion Convolution: P^k * X * W for forward/backward random walks
- Gated TCN: (X * W_f + b_f) ⊙ σ(X * W_g + b_g) with dilated convolutions
- Skip Connections: Residual + skip from each layer to output

**Key Benefits:**

- No need for predefined graph structure (adaptive learning)
- Captures long-range temporal dependencies via dilated convolutions
- Bidirectional spatial propagation captures complex graph patterns
- Efficient parallel training unlike recurrent models

## How It Works

GraphWaveNet combines graph convolution networks with WaveNet-style dilated causal
convolutions to capture both spatial and temporal dependencies in time series data.

**Reference:** Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling", IJCAI 2019.
https://arxiv.org/abs/1906.00121

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphWaveNetOptions` | Initializes a new instance of the `GraphWaveNetOptions` class with default values. |
| `GraphWaveNetOptions(GraphWaveNetOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionSteps` | Gets or sets the number of diffusion steps (K) for graph convolution. |
| `DilationChannels` | Gets or sets the number of dilation channels. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EndChannels` | Gets or sets the number of end (output) channels. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `LayersPerBlock` | Gets or sets the number of layers per temporal convolution block. |
| `NodeEmbeddingDim` | Gets or sets the dimension of node embeddings for adaptive graph learning. |
| `NumBlocks` | Gets or sets the number of temporal convolution blocks. |
| `NumFeatures` | Gets or sets the number of input features per node. |
| `NumNodes` | Gets or sets the number of nodes in the graph. |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `ResidualChannels` | Gets or sets the number of residual channels. |
| `SequenceLength` | Gets or sets the sequence length (input time steps). |
| `SkipChannels` | Gets or sets the number of skip connection channels. |
| `UseAdaptiveGraph` | Gets or sets whether to use adaptive graph learning. |
| `UsePredefinedGraph` | Gets or sets whether to also use a predefined adjacency matrix. |

