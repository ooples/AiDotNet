---
title: "Nougat<T>"
description: "Nougat neural network for academic document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.PixelToSequence`

Nougat neural network for academic document understanding.

## For Beginners

Nougat excels at:

1. Converting PDF pages to Markdown
2. Preserving mathematical equations (LaTeX)
3. Understanding document structure (sections, tables, figures)
4. Processing scientific papers without OCR

Example usage:

## How It Works

Nougat (Neural Optical Understanding for Academic Documents) is an OCR-free
transformer model specifically designed to parse academic papers from PDF images
into structured Markdown format with mathematical notation support.

**Reference:** "Nougat: Neural Optical Understanding for Academic Documents" (arXiv 2023)
https://arxiv.org/abs/2308.13418

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Nougat(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,NougatOptions)` | Creates a Nougat model using native layers for training and inference. |
| `Nougat(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,NougatOptions)` | Creates a Nougat model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportsLatex` | Gets whether this model supports LaTeX equation output. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies Nougat's industry-standard postprocessing: pass-through (Markdown outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies Nougat's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractEquations(Tensor<>)` | Parses a PDF page and extracts equations in LaTeX format. |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `ParseAcademicDocument(Tensor<>)` | Parses an academic document page and generates Markdown output. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

