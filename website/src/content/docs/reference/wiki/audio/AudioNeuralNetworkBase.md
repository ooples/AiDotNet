---
title: "AudioNeuralNetworkBase<T>"
description: "Base class for audio-focused neural networks that can operate in both ONNX inference and native training modes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio`

Base class for audio-focused neural networks that can operate in both ONNX inference and native training modes.

## For Beginners

Audio neural networks process sound data (like speech or music).
This base class provides:

- Support for pre-trained ONNX models (fast inference with existing models)
- Full training capability from scratch (like other neural networks)
- Audio preprocessing utilities (mel spectrograms, etc.)
- Sample rate handling

You can use this class in two ways:

1. Load a pre-trained ONNX model for quick inference
2. Build and train a new model from scratch

## How It Works

This class extends `NeuralNetworkBase` to provide audio-specific functionality
while maintaining full integration with the AiDotNet neural network infrastructure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioNeuralNetworkBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the AudioNeuralNetworkBase class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function for this model. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MelSpec` | Gets the mel spectrogram extractor for preprocessing. |
| `NumMels` | Gets the number of mel spectrogram channels used by this model. |
| `OnnxDecoder` | Gets or sets the ONNX decoder model (for encoder-decoder architectures). |
| `OnnxEncoder` | Gets or sets the ONNX encoder model (for encoder-decoder architectures). |
| `OnnxModel` | Gets or sets the ONNX model (for single-model architectures). |
| `SampleRate` | Gets the sample rate expected by this model. |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateMelSpectrogram(Int32,Int32,Int32,Int32)` | Creates a mel spectrogram extractor with the model's settings. |
| `Dispose(Boolean)` | Disposes of resources used by this model. |
| `Forward(Tensor<>)` | Performs a forward pass through the native neural network layers. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `RunOnnxInference(Tensor<>)` | Runs inference using ONNX model(s). |

## Fields

| Field | Summary |
|:-----|:--------|
| `TextEncoderLayers` | Text-encoder layer stack for dual-encoder audio-text models (CLAP and similar contrastive audio-language models). |

