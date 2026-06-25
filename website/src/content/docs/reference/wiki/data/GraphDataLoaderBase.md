---
title: "GraphDataLoaderBase<T>"
description: "Abstract base class for graph data loaders providing common graph-related functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Loaders`

Abstract base class for graph data loaders providing common graph-related functionality.

## For Beginners

This base class handles common graph operations:

- Storing node features and edge connections
- Creating different types of tasks (node classification, link prediction)
- Splitting data for training and evaluation

Concrete implementations (CitationNetworkLoader, MolecularDatasetLoader) extend this
to load specific graph datasets.

## How It Works

GraphDataLoaderBase provides shared implementation for all graph data loaders including:

- Node feature and adjacency matrix management
- Task creation (node classification, graph classification, link prediction)
- Train/validation/test mask generation
- Batch iteration for multiple graphs

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjacencyMatrix` |  |
| `BatchSize` |  |
| `EdgeIndex` |  |
| `GraphLabels` |  |
| `HasNext` |  |
| `NodeFeatures` |  |
| `NodeLabels` |  |
| `NumClasses` |  |
| `NumEdges` |  |
| `NumGraphs` |  |
| `NumNodeFeatures` |  |
| `NumNodes` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateGraphClassificationTask(Double,Double,Nullable<Int32>)` |  |
| `CreateLinkPredictionTask(Double,Double,Nullable<Int32>)` |  |
| `CreateNodeClassificationTask(Double,Double,Nullable<Int32>)` |  |
| `ExtractEdges(Tensor<>,Int32[])` | Extracts edges at specified indices from an edge index tensor. |
| `ExtractLabels(Tensor<>,Int32[])` | Extracts labels at specified indices from a label tensor. |
| `GenerateNegativeEdges(GraphData<>,Int32,Random)` | Generates negative edge samples (non-existing edges) as a tensor. |
| `GetBatches(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>)` |  |
| `GetBatchesAsync(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>,Int32,CancellationToken)` |  |
| `GetNextBatch` |  |
| `OnReset` |  |
| `TryGetNextBatch(GraphData<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `LoadedGraphData` | Storage for loaded graph data. |
| `LoadedGraphs` | Storage for multiple graphs (for graph classification datasets). |
| `NumOps` | Numeric operations helper for type T. |

