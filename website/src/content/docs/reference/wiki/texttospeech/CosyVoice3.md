---
title: "CosyVoice3<T>"
description: "CosyVoice 3: Fun-CosyVoice 3 zero-shot multilingual TTS with enhanced semantic tokens."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

CosyVoice 3: Fun-CosyVoice 3 zero-shot multilingual TTS with enhanced semantic tokens.

## For Beginners

CosyVoice 3 is an advanced version of CosyVoice that supports
zero-shot multilingual speech synthesis. It uses enhanced supervised semantic tokens and a
multi-scale flow matching decoder to produce natural-sounding speech in multiple languages
without needing training data for each target speaker.

## How It Works

**References:**

- Paper: "Fun-CosyVoice 3: Zero-Shot Multilingual TTS" (Alibaba DAMO, 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosyVoice3(NeuralNetworkArchitecture<>,CosyVoice3Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `CosyVoice3` class in native training/inference mode. |
| `CosyVoice3(NeuralNetworkArchitecture<>,String,CosyVoice3Options)` | Initializes a new instance of the `CosyVoice3` class in ONNX inference mode. |

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
| `Synthesize(String)` | Synthesizes speech from text using CosyVoice 3's enhanced pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

