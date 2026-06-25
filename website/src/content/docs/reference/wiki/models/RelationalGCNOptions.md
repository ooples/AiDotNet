---
title: "RelationalGCNOptions<T>"
description: "Configuration options for RelationalGCN (Relational Graph Convolutional Network)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for RelationalGCN (Relational Graph Convolutional Network).

## For Beginners

RelationalGCN is designed for knowledge graphs and multi-relational data:

**The Key Insight:**
Standard GCN treats all edges equally, but in financial networks, different types of
relationships matter differently. A "supplier-of" relationship is different from a
"competitor-of" relationship. R-GCN learns separate transformations for each relation type.

**What Problems Does RelationalGCN Solve?**

- Entity classification in knowledge graphs (company type, sector classification)
- Link prediction in multi-relational networks (predicting missing relationships)
- Financial network analysis with multiple relationship types
- Supply chain modeling with different connection types

**How RelationalGCN Works:**

1. **Relation-Specific Weights:** Learns different weights for each relation type
2. **Basis Decomposition:** Efficiently shares parameters across relations
3. **Block Decomposition:** Alternative parameter sharing using block-diagonal matrices
4. **Self-Connections:** Special weight for node's own features

**RelationalGCN Architecture:**

- For each relation r: H^(l+1) = sum_r (A_r * H^(l) * W_r) where A_r is the adjacency for relation r
- Basis decomposition: W_r = sum_b (a_rb * B_b) with shared bases B
- Block decomposition: W_r = diag(W_r1, W_r2, ..., W_rB) with smaller matrices

**Key Benefits:**

- Handles heterogeneous graphs with multiple edge types
- Parameter efficient through basis or block decomposition
- Captures relation-specific patterns in the data
- Effective for both entity classification and link prediction

## How It Works

RelationalGCN extends Graph Convolutional Networks to handle multi-relational data
where different types of edges (relations) exist between nodes.

**Reference:** Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks", ESWC 2018.
https://arxiv.org/abs/1703.06103

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RelationalGCNOptions` | Initializes a new instance of the `RelationalGCNOptions` class with default values. |
| `RelationalGCNOptions(RelationalGCNOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Aggregation` | Gets or sets the aggregation method for neighbor messages. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension for the model. |
| `NumBases` | Gets or sets the number of bases for basis decomposition. |
| `NumBlocks` | Gets or sets the number of blocks for block decomposition. |
| `NumFeatures` | Gets or sets the number of input features per node. |
| `NumLayers` | Gets or sets the number of R-GCN layers. |
| `NumNodes` | Gets or sets the number of nodes (entities) in the graph. |
| `NumRelations` | Gets or sets the number of relation types. |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `Regularization` | Gets or sets the regularization strength. |
| `SequenceLength` | Gets or sets the sequence length (input time steps). |
| `UseBasisDecomposition` | Gets or sets whether to use basis decomposition. |
| `UseBlockDecomposition` | Gets or sets whether to use block decomposition. |
| `UseSelfLoop` | Gets or sets whether to add self-loops to the graph. |

