---
title: "MTGNNOptions<T>"
description: "Configuration options for MTGNN (Multivariate Time-series Graph Neural Network)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for MTGNN (Multivariate Time-series Graph Neural Network).

## For Beginners

MTGNN is unique because it LEARNS how variables are connected:

**The Key Insight:**
Unlike other graph models that require you to define the graph structure upfront,
MTGNN automatically discovers which time series influence each other through
an adaptive graph learning module. This is powerful when relationships are unknown.

**What Problems Does MTGNN Solve?**

- Traffic prediction when road network is complex or unknown
- Multivariate financial forecasting with unknown correlations
- Sensor networks where dependencies change over time
- Any multivariate time series where inter-variable relationships matter

**How MTGNN Works:**

1. **Graph Learning:** Learns node embeddings and computes their similarity
2. **Mix-hop Propagation:** Aggregates information from different hop distances
3. **Dilated Inception:** Captures multi-scale temporal patterns
4. **Joint Learning:** Graph structure and predictions are learned together

**MTGNN Architecture:**

- Node Embeddings: Each node gets a learnable embedding vector
- Adaptive Adjacency: A = softmax(E1 * E2^T) computes learned graph
- Mix-hop Propagation: Combines 1-hop, 2-hop, ... K-hop neighbors
- Dilated Inception: Parallel dilated convolutions for temporal patterns

**Key Benefits:**

- No need to predefine graph structure
- Discovers hidden variable relationships
- Handles multivariate series with complex dependencies
- Combines best of GCN and TCN architectures

## How It Works

MTGNN is a graph neural network that simultaneously learns the graph structure
and performs spatio-temporal forecasting, without requiring a predefined adjacency matrix.

**Reference:** Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks", KDD 2020.
https://arxiv.org/abs/2005.11650

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MTGNNOptions` | Initializes a new instance of the `MTGNNOptions` class with default values. |
| `MTGNNOptions(MTGNNOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DilationFactor` | Gets or sets the dilation factor for temporal convolutions. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension for the model. |
| `MixHopDepth` | Gets or sets the depth of mix-hop propagation. |
| `NodeEmbeddingDim` | Gets or sets the dimension of node embeddings for graph learning. |
| `NumFeatures` | Gets or sets the number of features per node per time step. |
| `NumLayers` | Gets or sets the number of layers in the model. |
| `NumNodes` | Gets or sets the number of nodes (variables/time series). |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `SequenceLength` | Gets or sets the sequence length (input time steps). |
| `SubgraphSize` | Gets or sets the size of sampled subgraphs. |
| `TemporalKernelSize` | Gets or sets the kernel size for temporal convolutions. |
| `UsePredefinedGraph` | Gets or sets whether to use a predefined adjacency matrix alongside learned graph. |
| `UseSubgraphSampling` | Gets or sets whether to use subgraph sampling during training. |

