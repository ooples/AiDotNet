---
title: "SileroVad<T>"
description: "Silero Voice Activity Detection model - high accuracy neural network VAD."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.VoiceActivity`

Silero Voice Activity Detection model - high accuracy neural network VAD.

## For Beginners

Silero VAD tells you when someone is speaking vs silence.
Unlike simple energy-based VAD, it uses a neural network that has learned what
speech "looks like" from millions of examples.

Why use neural network VAD?

- Much more accurate than energy/threshold-based methods
- Handles background noise better (music, crowd noise, etc.)
- Detects speech even when quiet
- Doesn't false-trigger on non-speech sounds

Two ways to use this class:

1. ONNX Mode: Load pretrained Silero model for fast inference
2. Native Mode: Train your own VAD model from scratch

ONNX Mode Example (recommended):

Training Mode Example:

## How It Works

Silero VAD is a state-of-the-art voice activity detector that uses a lightweight
neural network architecture to achieve high accuracy with low latency. It can:

- Detect speech with very high accuracy (better than energy-based methods)
- Handle noisy environments well
- Run in real-time on CPU
- Work across multiple languages

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SileroVad(NeuralNetworkArchitecture<>,Int32,Int32,Double,Int32,Int32,Int32,Int32,Int32,SileroVadOptions)` | Creates a Silero VAD in native training mode for training from scratch. |
| `SileroVad(NeuralNetworkArchitecture<>,String,Int32,Int32,Double,Int32,Int32,SileroVadOptions)` | Creates a Silero VAD in ONNX inference mode with a pretrained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameSize` |  |
| `MinSilenceDurationMs` |  |
| `MinSpeechDurationMs` |  |
| `Threshold` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IVoiceActivityDetector{T}#ResetState` | Resets the VAD streaming state (implements IVoiceActivityDetector). |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectSpeech(Tensor<>)` |  |
| `DetectSpeechSegments(Tensor<>)` |  |
| `Dispose(Boolean)` |  |
| `ExtractLastTimestep(Tensor<>)` | Extracts the last timestep from a sequence tensor. |
| `ExtractLayerReferences` | (Re)populates the conv / LSTM / output sub-layer references from the canonical `Layers` list and materializes any lazy weights. |
| `Forward(Tensor<>)` |  |
| `ForwardForTraining(Tensor<>)` |  |
| `GetFrameProbabilities(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `GetSpeechProbability(Tensor<>)` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetVadState` | Resets the VAD streaming state. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_convFilters` | Number of convolutional filters. |
| `_convLayers` | Convolutional feature extraction layers. |
| `_disposed` | Disposed flag. |
| `_frameSize` | Frame size in samples. |
| `_inSpeech` | Current speech state. |
| `_lossFunction` | Loss function for training. |
| `_lstmHiddenDim` | LSTM hidden dimension. |
| `_lstmLayers` | LSTM layers for temporal modeling. |
| `_minSilenceDurationMs` | Minimum silence duration in milliseconds. |
| `_minSpeechDurationMs` | Minimum speech duration in milliseconds. |
| `_modelPath` | Path to the ONNX model file. |
| `_numLstmLayers` | Number of LSTM layers. |
| `_outputLayer` | Output classification layer. |
| `_silenceFrameCount` | Number of consecutive silence frames. |
| `_speechFrameCount` | Number of consecutive speech frames. |
| `_threshold` | Detection threshold (0-1). |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

