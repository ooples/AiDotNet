---
title: "TrOCR<T>"
description: "TrOCR (Transformer-based OCR) for text recognition from cropped images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextRecognition`

TrOCR (Transformer-based OCR) for text recognition from cropped images.

## For Beginners

TrOCR reads text from images. Given a cropped image of text
(like a single word or line), it outputs the actual characters. It works by:

1. The encoder (ViT) analyzes the image and creates feature representations
2. The decoder generates text one character at a time, using attention to focus on relevant image regions

Example usage:

## How It Works

TrOCR is an end-to-end text recognition model that uses a Vision Transformer (ViT)
encoder and a Transformer decoder (similar to BART/GPT-2) for sequence generation.

**Reference:** "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models" (AAAI 2022)
https://arxiv.org/abs/2109.10282

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrOCR(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,TrOCROptions)` | Creates a TrOCR model using native layers for training and inference. |
| `TrOCR(NeuralNetworkArchitecture<>,String,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,TrOCROptions)` | Creates a TrOCR model using pre-trained ONNX models for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AiDotNet#Document#Interfaces#ITextRecognizer{T}#MaxSequenceLength` |  |
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedCharacters` |  |
| `SupportedDocumentTypes` |  |
| `SupportsAttentionVisualization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies TrOCR's industry-standard postprocessing: pass-through (encoder-decoder outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies TrOCR's industry-standard preprocessing: normalize to [-1, 1]. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `GetAttentionWeights` |  |
| `GetCharacterProbabilities` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `RecognizeText(Tensor<>)` |  |
| `RecognizeTextBatch(IEnumerable<Tensor<>>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

