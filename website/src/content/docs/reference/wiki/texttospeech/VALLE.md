---
title: "VALLE<T>"
description: "VALL-E: neural codec language model for zero-shot text-to-speech using autoregressive and non-autoregressive transformers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

VALL-E: neural codec language model for zero-shot text-to-speech using autoregressive and non-autoregressive transformers.

## For Beginners

VALL-E treats text-to-speech as a language modeling problem with audio tokens.
Given just 3 seconds of a person's voice as a prompt, it can generate speech in that person's voice
saying any text. It works in two stages: first predicting coarse audio tokens autoregressively (one at a time),
then filling in fine detail tokens all at once (non-autoregressively).

## How It Works

**References:**

- Paper: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (Wang et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VALLE(NeuralNetworkArchitecture<>,String,VALLEOptions)` | Initializes a new instance of the `VALLE` class in ONNX inference mode. |
| `VALLE(NeuralNetworkArchitecture<>,VALLEOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `VALLE` class in native training/inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DecodeFromTokens(Tensor<>)` | Decodes discrete codec tokens back into audio. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeToTokens(Tensor<>)` | Encodes audio into discrete codec tokens. |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessAudio(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessText(String)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Synthesize(String)` | Synthesizes speech using VALL-E's two-stage codec language model. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

