---
title: "LayoutGraph<T>"
description: "LayoutGraph for graph-based document layout analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.GraphBased`

LayoutGraph for graph-based document layout analysis.

## For Beginners

LayoutGraph analyzes how document parts relate:

1. Builds a graph from document structure
2. Models reading order and containment
3. Learns hierarchical relationships
4. Predicts document element types and groupings

Key features:

- Hierarchical graph construction
- Spatial relationship modeling
- Reading order prediction
- Multi-level layout understanding

Example usage:

## How It Works

LayoutGraph constructs and analyzes graphs from document layouts, where nodes
represent document elements and edges encode spatial relationships. It excels
at understanding hierarchical document structures.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayoutGraph(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutGraphOptions)` | Creates a LayoutGraph model using native layers for training and inference. |
| `LayoutGraph(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutGraphOptions)` | Creates a LayoutGraph model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `NodeDim` | Gets the node dimension. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies LayoutGraph's industry-standard postprocessing: pass-through (graph node classifications are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies LayoutGraph's industry-standard preprocessing: simple normalization to [0,1]. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `DetectReadingOrder(DocumentLayoutResult<>)` |  |
| `DetectReadingOrder(Tensor<>)` |  |
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

