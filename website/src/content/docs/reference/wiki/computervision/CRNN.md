---
title: "CRNN<T>"
description: "CRNN (Convolutional Recurrent Neural Network) for text recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.OCR.Recognition`

CRNN (Convolutional Recurrent Neural Network) for text recognition.

## For Beginners

CRNN combines CNNs for visual feature extraction with
RNNs (specifically LSTM) for sequence modeling. It's trained with CTC loss to
handle variable-length text without requiring character-level alignment.

## How It Works

Key features:

- CNN backbone for visual features
- Bidirectional LSTM for sequence modeling
- CTC (Connectionist Temporal Classification) decoding
- Handles variable-length text naturally

Reference: Shi et al., "An End-to-End Trainable Neural Network for
Image-based Sequence Recognition and Its Application to Scene Text Recognition", TPAMI 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CRNN(OCROptions<>)` | Creates a new CRNN text recognizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBatchNorm(Tensor<>)` | Applies simple batch normalization. |
| `ApplyBidirectionalLSTM(Tensor<>,Int32)` | Applies bidirectional LSTM using proper LSTMLayer cells. |
| `ApplySoftmax(Tensor<>)` | Applies softmax normalization across the vocabulary dimension. |
| `ConcatenateBidirectional(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Concatenates forward and backward LSTM outputs. |
| `ConvertToGrayscale(Tensor<>)` | Converts RGB image to grayscale. |
| `ExtractTimestep(Tensor<>,Int32,Int32,Int32)` | Extracts a single timestep from the sequence tensor. |
| `GetParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `LoadWeightsFromFile(String)` | Loads weights from a file. |
| `MapWeightsToLayers(Dictionary<String,Tensor<Single>>)` | Maps loaded weights to the model's layers. |
| `Recognize(Tensor<>)` |  |
| `RecognizeText(Tensor<>)` |  |
| `ResetLSTMStates(Int32)` | Resets the LSTM hidden and cell states. |
| `SaveWeights(String)` |  |
| `StoreTimestep(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Stores LSTM output into the sequence tensor at a specific timestep. |

