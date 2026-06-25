---
title: "VALLEX<T>"
description: "VALL-E X: cross-lingual zero-shot text-to-speech extending VALL-E with language ID conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

VALL-E X: cross-lingual zero-shot text-to-speech extending VALL-E with language ID conditioning.

## For Beginners

VALL-E X extends the original VALL-E model to work across different languages.
Given a short 3-second recording of someone speaking in one language, it can generate that person's
voice speaking in a completely different language. It achieves this by adding language ID conditioning
to VALL-E's autoregressive and non-autoregressive transformer stages, enabling cross-lingual
voice cloning without parallel training data.

## How It Works

**References:**

- Paper: "VALL-E X: Speak Foreign Languages with Your Own Voice" (Zhang et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VALLEX(NeuralNetworkArchitecture<>,String,VALLEXOptions)` | Initializes a new instance of the `VALLEX` class in ONNX inference mode. |
| `VALLEX(NeuralNetworkArchitecture<>,VALLEXOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `VALLEX` class in native training/inference mode. |

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
| `Synthesize(String)` | Synthesizes cross-lingual speech using VALL-E X's language-conditioned codec language model. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

