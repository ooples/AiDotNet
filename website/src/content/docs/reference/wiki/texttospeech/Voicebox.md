---
title: "Voicebox<T>"
description: "Voicebox: text-guided multilingual universal speech generation at scale using non-autoregressive flow matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.CodecBased`

Voicebox: text-guided multilingual universal speech generation at scale using non-autoregressive flow matching.

## For Beginners

Voicebox is a versatile speech generation model that can do much more than
just text-to-speech. It uses a technique called "flow matching" with an infilling objective, meaning
it learns to fill in missing parts of speech given surrounding context. This allows it to perform
tasks like noise removal, content editing (changing words in recorded speech), style transfer,
and cross-lingual speech generation. Unlike autoregressive models that generate speech one piece
at a time, Voicebox generates all parts simultaneously, making it faster and more flexible.

## How It Works

**References:**

- Paper: "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale" (Le et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Voicebox(NeuralNetworkArchitecture<>,String,VoiceboxOptions)` | Initializes a new instance of the `Voicebox` class in ONNX inference mode. |
| `Voicebox(NeuralNetworkArchitecture<>,VoiceboxOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `Voicebox` class in native training/inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DecodeFromTokens(Tensor<>)` | Decodes discrete codec tokens back into an audio waveform. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeToTokens(Tensor<>)` | Encodes audio into discrete codec tokens using frame-level quantization. |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessAudio(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessText(String)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Synthesize(String)` | Synthesizes speech from text using Voicebox's non-autoregressive flow matching pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

