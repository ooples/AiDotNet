---
title: "ABINet<T>"
description: "ABINet (Autonomous, Bidirectional, Iterative Network) for text recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextRecognition`

ABINet (Autonomous, Bidirectional, Iterative Network) for text recognition.

## For Beginners

ABINet has three key innovations:

1. Autonomous vision model (works without external language model)
2. Bidirectional language model (looks at context from both directions)
3. Iterative correction (refines predictions multiple times)

Key features:

- Self-contained (no external LM needed)
- Built-in spell correction via language model
- Iterative refinement for accuracy
- Strong on noisy/occluded text

Example usage:

## How It Works

ABINet uses a novel architecture with autonomous vision, bidirectional language modeling,
and iterative correction to achieve robust text recognition.

**Reference:** "Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling" (CVPR 2021)
https://arxiv.org/abs/2103.06495

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ABINet(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ABINetOptions)` | Creates an ABINet model using native layers for training and inference. |
| `ABINet(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ABINetOptions)` | Creates an ABINet model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `ImageHeight` | Gets the input image height. |
| `MaxSequenceLength` |  |
| `NumIterations` | Gets the number of iterative refinement steps. |
| `RequiresOCR` |  |
| `SupportedCharacters` |  |
| `SupportedDocumentTypes` |  |
| `SupportsAttentionVisualization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies ABINet's industry-standard postprocessing: pass-through (language model outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies ABINet's industry-standard preprocessing: text image preprocessing. |
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

