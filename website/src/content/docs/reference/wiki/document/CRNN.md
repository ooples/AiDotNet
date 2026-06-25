---
title: "CRNN<T>"
description: "CRNN (Convolutional Recurrent Neural Network) for sequence-based text recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextRecognition`

CRNN (Convolutional Recurrent Neural Network) for sequence-based text recognition.

## For Beginners

CRNN works by:

1. CNN extracts visual features from the text image
2. BiLSTM models the sequence of features
3. CTC decoding converts outputs to text

Key advantages:

- No need to segment individual characters
- Handles variable-length text
- End-to-end trainable
- Works with horizontal text lines

Example usage:

## How It Works

CRNN combines CNN for image feature extraction with RNN (BiLSTM) for sequence modeling,
trained with CTC loss for variable-length text recognition without explicit character
segmentation.

**Reference:** "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (TPAMI 2017)
https://arxiv.org/abs/1507.05717

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CRNN(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,CRNNOptions)` | Creates a CRNN model using native layers for training and inference. |
| `CRNN(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,CRNNOptions)` | Creates a CRNN model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `ImageHeight` | Gets the input image height expected by the model. |
| `MaxSequenceLength` |  |
| `RequiresOCR` |  |
| `SupportedCharacters` |  |
| `SupportedDocumentTypes` |  |
| `SupportsAttentionVisualization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies CRNN's industry-standard postprocessing: pass-through (CTC outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies CRNN's industry-standard preprocessing: text image preprocessing. |
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

