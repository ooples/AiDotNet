---
title: "Kokoro<T>"
description: "Kokoro: lightweight end-to-end TTS with a StyleTTS2-inspired architecture using style tokens and ISTFTNet decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.EndToEnd`

Kokoro: lightweight end-to-end TTS with a StyleTTS2-inspired architecture using style tokens and ISTFTNet decoder.

## For Beginners

Kokoro is a remarkably small but high-quality TTS model with only 82 million
parameters (compared to billions in larger models). It is inspired by StyleTTS2's architecture and uses
several clever techniques to achieve quality speech from a compact model:
(1) A BERT-style phoneme encoder processes text into hidden states,
(2) A style encoder predicts voice characteristics directly from text (no reference audio needed),
(3) A duration predictor determines how long each sound should last,
(4) An ISTFTNet decoder converts features into audio by predicting STFT magnitude and phase
and applying inverse STFT, which is faster than traditional waveform generation.
It supports 9 languages and runs in real-time on CPU.

## How It Works

**References:**

- Project: "Kokoro: A frontier TTS model for its size of 82M params" (Hexgrad, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Kokoro(NeuralNetworkArchitecture<>,KokoroOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `Kokoro` class in native training/inference mode. |
| `Kokoro(NeuralNetworkArchitecture<>,String,KokoroOptions)` | Initializes a new instance of the `Kokoro` class in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimension size. |
| `NumFlowSteps` | Gets the number of normalizing flow steps used in the decoder. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessAudio(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessText(String)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Synthesize(String)` | Synthesizes speech using Kokoro's StyleTTS2-inspired pipeline with ISTFTNet decoder. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

