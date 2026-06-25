---
title: "MATCHA<T>"
description: "MATCHA (Math-Aware Transformer for Chart Harvesting and Analysis) for chart understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.PixelToSequence`

MATCHA (Math-Aware Transformer for Chart Harvesting and Analysis) for chart understanding.

## For Beginners

MATCHA specializes in understanding charts:

1. Reads bar charts, line graphs, pie charts, scatter plots
2. Extracts underlying numerical data
3. Answers questions about chart content
4. Generates summaries of chart insights

Key features:

- Math-aware pre-training for numerical reasoning
- Pix2Struct-based architecture
- Chart derendering (image to data table)
- Chart QA and summarization

Example usage:

## How It Works

MATCHA is designed specifically for understanding charts and plots, combining
math-aware pre-training with visual encoding to extract data, answer questions,
and summarize chart content.

**Reference:** "MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering" (ACL 2023)
https://arxiv.org/abs/2212.09662

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MATCHA(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,MATCHAOptions)` | Creates a MATCHA model using native layers for training and inference. |
| `MATCHA(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,MATCHAOptions)` | Creates a MATCHA model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `MaxPatchesPerImage` | Gets the maximum patches per image. |
| `RequiresOCR` |  |
| `SupportedChartTypes` | Gets the supported chart types. |
| `SupportedDocumentTypes` |  |
| `SupportsBorderedTables` |  |
| `SupportsBorderlessTables` |  |
| `SupportsMergedCells` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies MATCHA's industry-standard postprocessing: pass-through (chart QA outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies MATCHA's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Converts token IDs to text using character-level decoding. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectTables(Tensor<>)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExportTables(Tensor<>,TableExportFormat)` |  |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `ExtractTableContent(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `RecognizeStructure(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

