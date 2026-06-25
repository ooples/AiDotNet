---
title: "CRNNEventDetector<T>"
description: "CRNN (Convolutional Recurrent Neural Network) model for Sound Event Detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

CRNN (Convolutional Recurrent Neural Network) model for Sound Event Detection.

## For Beginners

CRNN is like a two-stage sound detective:

Stage 1 (CNN): Looks at short moments of the spectrogram and identifies spectral patterns.
"This frequency pattern looks like a door slam" or "These harmonics look like speech."

Stage 2 (RNN): Reads those patterns over time like a story.
"The door slam pattern appeared briefly at 2.3 seconds and ended at 2.5 seconds."

Together, they can tell you WHAT sounds happened and WHEN they happened.

**Usage:**

## How It Works

CRNN for SED (Cakir et al., 2017) is the standard baseline architecture for the DCASE
Sound Event Detection challenge. It combines CNN layers for spectral feature extraction
with bidirectional GRU/LSTM layers for temporal modeling, producing frame-level event
probabilities. Despite its simplicity compared to Transformer-based models, CRNN remains
competitive and is widely used as a strong baseline.

**Architecture:**

- **CNN blocks**: 3 convolutional blocks (64/128/256 channels) with batch normalization,

ReLU activation, and frequency-axis pooling. Extracts local spectro-temporal features.

- **RNN layers**: 2 bidirectional GRU layers (128 hidden units each) that model

temporal dependencies across frames. Bidirectional processing captures both past and future context.

- **Classification head**: Frame-level linear projection with sigmoid activation

for multi-label detection (multiple events can occur simultaneously).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CRNNEventDetector(NeuralNetworkArchitecture<>,CRNNEventDetectorOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a CRNN SED model for native training mode. |
| `CRNNEventDetector(NeuralNetworkArchitecture<>,String,CRNNEventDetectorOptions)` | Creates a CRNN SED model for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` |  |
| `SupportedEvents` |  |
| `TimeResolution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` |  |
| `Detect(Tensor<>,)` |  |
| `DetectAsync(Tensor<>,CancellationToken)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` |  |
| `GetEventProbabilities(Tensor<>)` |  |
| `StartStreamingSession` |  |
| `StartStreamingSession(Int32,)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `AudioSetLabels` | AudioSet-527 standard event labels. |

