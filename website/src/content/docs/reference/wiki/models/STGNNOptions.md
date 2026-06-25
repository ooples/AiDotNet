---
title: "STGNNOptions<T>"
description: "Configuration options for STGNN (Spatio-Temporal Graph Neural Network)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for STGNN (Spatio-Temporal Graph Neural Network).

## For Beginners

STGNN combines two types of learning for forecasting:

**The Key Insight:**
Many time series are not independent - they're connected in space. Traffic at one
intersection affects nearby intersections; stock prices affect related stocks.
STGNN models both the spatial connections (graph) and temporal patterns (time series).

**What is a Spatio-Temporal Graph?**

- Nodes: Locations or entities (sensors, stocks, cities)
- Edges: Connections between nodes (roads, correlations, trade routes)
- Node Features: Time series at each node
- The graph captures "who affects whom"

**How STGNN Works:**

1. **Spatial Aggregation:** Each node gathers information from its neighbors
2. **Temporal Modeling:** Process the time series at each node
3. **Spatio-Temporal Fusion:** Combine spatial and temporal information
4. **Prediction:** Forecast future values for all nodes

**STGNN Architecture:**

- Graph Convolution: Aggregates neighbor information weighted by edge strength
- Temporal Convolution: Captures patterns in time using 1D convolutions
- Gated Mechanism: Controls information flow between spatial and temporal
- Skip Connections: Preserves input information through deep networks

**Key Benefits:**

- Models complex spatial dependencies
- Captures multi-scale temporal patterns
- Handles irregular graph structures
- Scalable to large networks

## How It Works

STGNN is a graph neural network designed for spatio-temporal forecasting that captures
both spatial dependencies (between nodes/locations) and temporal dynamics.

**Reference:** Yu et al., "Spatio-Temporal Graph Convolutional Networks", IJCAI 2018.
https://arxiv.org/abs/1709.04875

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STGNNOptions` | Initializes a new instance of the `STGNNOptions` class with default values. |
| `STGNNOptions(STGNNOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `GraphConvType` | Gets or sets the type of graph convolution to use. |
| `HiddenDimension` | Gets or sets the hidden dimension for graph and temporal convolutions. |
| `NumFeatures` | Gets or sets the number of features per node. |
| `NumNodes` | Gets or sets the number of nodes in the graph. |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `NumSpatialLayers` | Gets or sets the number of spatial (graph convolution) layers. |
| `NumTemporalLayers` | Gets or sets the number of temporal convolution layers. |
| `SequenceLength` | Gets or sets the sequence length (input time steps). |
| `TemporalKernelSize` | Gets or sets the kernel size for temporal convolutions. |
| `UseGatedFusion` | Gets or sets whether to use gated fusion of spatial and temporal features. |
| `UseResidualConnections` | Gets or sets whether to use residual (skip) connections. |

