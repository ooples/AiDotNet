---
title: "TableTransformer<T>"
description: "TableTransformer for table detection and structure recognition using DETR-style architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.Analysis.TableDetection`

TableTransformer for table detection and structure recognition using DETR-style architecture.

## For Beginners

TableTransformer helps computers understand tables in documents.
It can:

1. Find where tables are located in a page (table detection)
2. Identify the structure within tables - rows, columns, and cells (structure recognition)
3. Handle both bordered and borderless tables

Example usage:

## How It Works

TableTransformer is based on the DETR (DEtection TRansformer) architecture, adapted for
table detection and table structure recognition. It can detect tables in documents and
identify their internal structure (rows, columns, cells, headers).

**Reference:** "PubTables-1M: Towards Comprehensive Table Extraction from Unstructured Documents" (CVPR 2022)
https://arxiv.org/abs/2110.00061

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TableTransformer(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,TableTransformerOptions)` | Creates a TableTransformer model using native layers for training and inference. |
| `TableTransformer(NeuralNetworkArchitecture<>,String,String,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,TableTransformerOptions)` | Creates a TableTransformer model using pre-trained ONNX models for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `NumQueries` | Gets the number of object queries used in DETR decoder. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportsBorderedTables` |  |
| `SupportsBorderlessTables` |  |
| `SupportsMergedCells` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies TableTransformer's industry-standard postprocessing: pass-through (DETR outputs are already in final format). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies TableTransformer's industry-standard preprocessing: COCO/ImageNet normalization. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectTables(Tensor<>)` |  |
| `DetectTables(Tensor<>,Double)` | Detects tables with a custom confidence threshold. |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExportTables(Tensor<>,TableExportFormat)` |  |
| `ExtractTableContent(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `RecognizeStructure(Tensor<>)` |  |
| `RecognizeStructure(Tensor<>,Double)` | Recognizes table structure with a custom confidence threshold. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

