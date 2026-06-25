---
title: "DocGCN<T>"
description: "DocGCN (Document Graph Convolutional Network) for document understanding using graph neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.GraphBased`

DocGCN (Document Graph Convolutional Network) for document understanding using graph neural networks.

## For Beginners

DocGCN views documents as networks:

1. Each text block becomes a node in a graph
2. Nearby blocks are connected by edges
3. Graph convolutions learn relationships
4. Can classify, extract, or understand document structure

Key features:

- Graph-based document representation
- Spatial relationship modeling
- Multi-hop reasoning through graph layers
- Entity and relation extraction

Example usage:

## How It Works

DocGCN represents documents as graphs where nodes are text blocks and edges represent
spatial and semantic relationships. Graph convolutional layers propagate information
to understand document structure.

**Reference:** Based on graph neural network approaches for document understanding.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocGCN` | Creates a DocGCN model with default configuration for native training. |
| `DocGCN(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocGCNOptions)` | Creates a DocGCN model using native layers for training and inference. |
| `DocGCN(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocGCNOptions)` | Creates a DocGCN model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `MaxNodes` | Gets the maximum number of nodes. |
| `NodeDim` | Gets the node feature dimension. |
| `NumGCNLayers` | Gets the number of GCN layers. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies DocGCN's industry-standard postprocessing: pass-through (node classifications are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies DocGCN's industry-standard preprocessing: simple normalization to [0,1]. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

