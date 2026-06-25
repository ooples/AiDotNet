---
title: "DocBank<T>"
description: "DocBank model for document page segmentation and layout analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.Analysis.PageSegmentation`

DocBank model for document page segmentation and layout analysis.

## For Beginners

DocBank divides document pages into different regions:

- Paragraphs: Regular text content
- Titles: Document headings and titles
- Figures: Images and diagrams
- Tables: Tabular data regions
- Captions: Text describing figures/tables
- Lists: Bulleted or numbered lists
- Equations: Mathematical formulas
- And more...

Example usage:

## How It Works

DocBank is a benchmark and model for document layout analysis that can segment document
pages into semantic regions including text, titles, figures, tables, and captions.
It combines visual features with optional text features for robust segmentation.

**Reference:** "DocBank: A Benchmark Dataset for Document Layout Analysis" (COLING 2020)
https://arxiv.org/abs/2006.01038

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocBank(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Boolean,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocBankOptions)` | Creates a DocBank model using native layers for training and inference. |
| `DocBank(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Boolean,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocBankOptions)` | Creates a DocBank model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `NumClasses` | Gets the number of segmentation classes. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedRegionTypes` |  |
| `SupportsInstanceSegmentation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies DocBank's industry-standard postprocessing: softmax over class dimension. |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies DocBank's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `GetSegmentationMask(Tensor<>)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SegmentPage(Tensor<>)` |  |
| `SegmentPage(Tensor<>,Double)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

