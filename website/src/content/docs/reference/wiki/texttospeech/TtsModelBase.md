---
title: "TtsModelBase<T>"
description: "Base class for text-to-speech neural networks that can operate in both ONNX inference and native training modes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.TextToSpeech`

Base class for text-to-speech neural networks that can operate in both ONNX inference and native training modes.

## For Beginners

Text-to-speech models convert written text into spoken audio. This base class provides:

- Support for pre-trained ONNX models (fast inference with existing models)
- Full training capability from scratch (like other neural networks)
- Audio preprocessing utilities (mel-spectrogram computation, normalization)
- Text encoding utilities (phoneme/token conversion)

You can use this class in two ways:

1. Load a pre-trained ONNX model for quick inference
2. Build and train a new model from scratch

## How It Works

This class extends `NeuralNetworkBase` to provide TTS-specific functionality
while maintaining full integration with the AiDotNet neural network infrastructure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TtsModelBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the TtsModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function for this model. |
| `HiddenDim` | Gets the model's hidden dimension. |
| `HopSize` | Gets the hop size in audio samples for mel-spectrogram computation. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MelChannels` | Gets the number of mel-spectrogram frequency channels. |
| `OnnxDecoder` | Gets or sets the ONNX decoder model (for two-stage architectures). |
| `OnnxEncoder` | Gets or sets the ONNX encoder model (for two-stage architectures). |
| `OnnxModel` | Gets or sets the ONNX model (for single-model architectures). |
| `SampleRate` | Gets the audio sample rate in Hz. |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose(Boolean)` | Disposes of resources used by this model. |
| `Gelu(Double)` | Applies GELU activation function element-wise. |
| `GetOrCreateBaseOptimizer` | Vocoder generators (those implementing `IVocoder`) train with AMSGrad rather than plain Adam. |
| `L2Normalize(Tensor<>)` | L2-normalizes a tensor. |
| `NormalizeMel(Tensor<>,Double,Double)` | Normalizes a mel-spectrogram tensor. |
| `PostprocessAudio(Tensor<>)` | Postprocesses model output into the final audio format. |
| `PreprocessText(String)` | Preprocesses raw text into a token tensor for model input. |
| `Softmax(Tensor<>)` | Applies softmax to convert logits to probabilities. |

