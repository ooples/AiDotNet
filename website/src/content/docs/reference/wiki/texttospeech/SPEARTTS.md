---
title: "SPEARTTS<T>"
description: "SPEAR-TTS: high-fidelity text-to-speech with minimal supervision using a speak-read-prompt pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

SPEAR-TTS: high-fidelity text-to-speech with minimal supervision using a speak-read-prompt pipeline.

## For Beginners

SPEAR-TTS generates speech by first converting text into semantic tokens
(high-level representations of speech content), then converting those into acoustic tokens
(detailed audio features using SoundStream codes). This two-stage approach allows the model
to produce high-quality speech with very little labeled training data.

## How It Works

**References:**

- Paper: "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision" (Kharitonov et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SPEARTTS(NeuralNetworkArchitecture<>,SPEARTTSOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `SPEARTTS` class in native training/inference mode. |
| `SPEARTTS(NeuralNetworkArchitecture<>,String,SPEARTTSOptions)` | Initializes a new instance of the `SPEARTTS` class in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DecodeFromTokens(Tensor<>)` | Decodes discrete codec tokens back into an audio waveform. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeToTokens(Tensor<>)` | Encodes audio into discrete codec tokens using SoundStream-style quantization. |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessAudio(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessText(String)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Synthesize(String)` | Synthesizes speech from text using SPEAR-TTS's semantic-to-acoustic pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

