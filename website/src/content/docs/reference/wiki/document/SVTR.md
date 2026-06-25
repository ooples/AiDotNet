---
title: "SVTR<T>"
description: "SVTR (Scene Text Visual Transformer) for text recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextRecognition`

SVTR (Scene Text Visual Transformer) for text recognition.

## For Beginners

SVTR modernizes text recognition:

1. Uses vision transformer (no RNN needed)
2. Handles various text heights and lengths
3. Multi-scale feature extraction
4. Efficient single-stream architecture

Key features:

- Pure transformer architecture
- Local + global mixing blocks
- Height compression for efficiency
- State-of-the-art accuracy

Example usage:

## How It Works

SVTR is a single-stream vision transformer for scene text recognition that processes
text images as visual sequences without requiring recurrent networks.

**Reference:** "SVTR: Scene Text Recognition with a Single Visual Model" (IJCAI 2022)
https://arxiv.org/abs/2205.00159

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SVTR` | Creates an SVTR model with default configuration for native training. |
| `SVTR(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,SVTROptions)` | Creates an SVTR model using native layers for training and inference. |
| `SVTR(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,SVTROptions)` | Creates an SVTR model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `ImageHeight` | Gets the input image height. |
| `MaxSequenceLength` |  |
| `RequiresOCR` |  |
| `SupportedCharacters` |  |
| `SupportedDocumentTypes` |  |
| `SupportsAttentionVisualization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies SVTR's industry-standard postprocessing: pass-through (transformer outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies SVTR's industry-standard preprocessing: text image preprocessing. |
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

