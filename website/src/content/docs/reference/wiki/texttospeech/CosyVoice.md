---
title: "CosyVoice<T>"
description: "CosyVoice: multilingual TTS with supervised semantic tokens and conditional flow matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

CosyVoice: multilingual TTS with supervised semantic tokens and conditional flow matching.

## For Beginners

CosyVoice is a multilingual text-to-speech model that can clone voices
without any training on the target speaker (zero-shot). It works by first converting text into
"semantic tokens" using a language model, then using conditional flow matching to generate
high-quality mel-spectrograms from those tokens.

## How It Works

**References:**

- Paper: "CosyVoice: A Scalable Multilingual Zero-Shot Text-to-Speech Synthesizer Based on Supervised Semantic Tokens" (Du et al., 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosyVoice(NeuralNetworkArchitecture<>,CosyVoiceOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `CosyVoice` class in native training/inference mode. |
| `CosyVoice(NeuralNetworkArchitecture<>,String,CosyVoiceOptions)` | Initializes a new instance of the `CosyVoice` class in ONNX inference mode. |

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
| `Synthesize(String)` | Synthesizes speech from text using CosyVoice's pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

