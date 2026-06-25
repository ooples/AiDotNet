---
title: "UDOP<T>"
description: "UDOP (Unifying Vision, Text, and Layout for Universal Document Processing) neural network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.VisionLanguage`

UDOP (Unifying Vision, Text, and Layout for Universal Document Processing) neural network.

## For Beginners

UDOP can handle many document tasks with one model:

1. Document classification
2. Information extraction (NER, key-value pairs)
3. Document question answering
4. Document layout analysis
5. Document generation

Example usage:

## How It Works

UDOP is a foundation model for document AI that unifies text, image, and layout modalities
within a single encoder-decoder framework. It can perform multiple document tasks through
task-specific prompting.

**Reference:** "Unifying Vision, Text, and Layout for Universal Document Processing" (CVPR 2023)
https://arxiv.org/abs/2212.02623

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UDOP(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,UDOPOptions)` | Creates a UDOP model using native layers for training and inference. |
| `UDOP(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,UDOPOptions)` | Creates a UDOP model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableCategories` | Gets the available document classification categories. |
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies UDOP's industry-standard postprocessing: pass-through (unified outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies UDOP's industry-standard preprocessing: ImageNet normalization. |
| `ClassifyDocument(Tensor<>)` |  |
| `ClassifyDocument(Tensor<>,Int32)` |  |
| `CreateNewInstance` |  |
| `DecodeGenerativeOutput(Tensor<>,Int32)` | Decodes generative output from UDOP model. |
| `DecodeTokensToText(List<Int32>)` | Decodes token IDs to text using T5-style vocabulary. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `Forward(Tensor<>)` | Encoder-decoder forward pass: runs encoder layers, then feeds encoder output as cross-attention context to decoder layers. |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

