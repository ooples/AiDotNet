---
title: "AudioLM<T>"
description: "AudioLM: language modeling approach to audio generation with semantic and acoustic tokens."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

AudioLM: language modeling approach to audio generation with semantic and acoustic tokens.

## For Beginners

AudioLM treats audio generation as a language modeling problem.
Instead of predicting words, it predicts audio tokens. It first generates high-level "semantic" tokens
that capture the meaning and content of speech, then generates detailed "acoustic" tokens that add
the fine details needed for natural-sounding audio.

## How It Works

**References:**

- Paper: "AudioLM: A Language Modeling Approach to Audio Generation" (Borsos et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioLM(NeuralNetworkArchitecture<>,AudioLMOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `AudioLM` class in native training/inference mode. |
| `AudioLM(NeuralNetworkArchitecture<>,String,AudioLMOptions)` | Initializes a new instance of the `AudioLM` class in ONNX inference mode. |

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
| `Synthesize(String)` | Synthesizes speech using AudioLM's hierarchical token generation pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

