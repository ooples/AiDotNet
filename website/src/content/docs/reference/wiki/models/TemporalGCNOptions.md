---
title: "TemporalGCNOptions<T>"
description: "Configuration options for TemporalGCN (Temporal Graph Convolutional Network)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TemporalGCN (Temporal Graph Convolutional Network).

## For Beginners

TemporalGCN captures two types of patterns simultaneously:

**The Key Insight:**
Many real-world systems have both spatial structure (like a road network) and
temporal dynamics (like traffic patterns over time). TemporalGCN learns both
by alternating between spatial and temporal processing layers.

**What is Temporal Graph Convolution?**

- Standard GCN: Aggregates information from neighboring nodes at one time step
- TemporalGCN: Also captures how patterns evolve over time
- Combines spatial GCN layers with temporal recurrent (GRU/LSTM) layers
- Creates a "video" view of the graph, not just a "snapshot"

**How TemporalGCN Works:**

1. **Graph Convolution:** Each node aggregates features from neighbors
2. **Temporal Recurrence:** GRU/LSTM processes the sequence at each node
3. **Spatial-Temporal Stacking:** Alternate spatial and temporal layers
4. **Prediction:** Output future values at each node

**TemporalGCN Architecture:**

- GCN layers with Chebyshev polynomial approximation
- GRU cells for temporal modeling
- Batch normalization for stability
- Residual connections for gradient flow

**Key Benefits:**

- Jointly learns spatial and temporal patterns
- Handles dynamic graphs where edges change
- Scales to large graphs with sparse operations
- Works with irregular spatial structures

## How It Works

TemporalGCN combines graph convolutional networks with recurrent neural networks
to model both spatial and temporal dependencies in graph-structured time series data.

**Reference:** Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction", 2019.
https://arxiv.org/abs/1811.05320

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalGCNOptions` | Initializes a new instance of the `TemporalGCNOptions` class with default values. |
| `TemporalGCNOptions(TemporalGCNOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChebyshevOrder` | Gets or sets the Chebyshev polynomial order for graph convolution. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension for GCN and temporal layers. |
| `NumFeatures` | Gets or sets the number of features per node. |
| `NumGCNLayers` | Gets or sets the number of GCN layers. |
| `NumNodes` | Gets or sets the number of nodes in the graph. |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `NumTemporalLayers` | Gets or sets the number of temporal (recurrent) layers. |
| `SequenceLength` | Gets or sets the sequence length (input time steps). |
| `TemporalCellType` | Gets or sets the type of recurrent cell for temporal modeling. |
| `UseBatchNormalization` | Gets or sets whether to use batch normalization. |
| `UseResidualConnections` | Gets or sets whether to use residual (skip) connections. |

