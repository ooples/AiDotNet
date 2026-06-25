---
title: "ParlerTTS<T>"
description: "Parler-TTS: text-described TTS that generates speech matching a natural language voice description."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.DescriptionBased`

Parler-TTS: text-described TTS that generates speech matching a natural language voice description.

## For Beginners

Parler-TTS takes a unique approach to controlling speech output: instead of
needing a reference audio clip of a speaker, you simply describe the voice you want in plain English.
For example, you might say "A warm female voice with a slight British accent, speaking slowly and clearly"
and the model will generate speech matching that description. It works by using a text encoder for the
voice description, a DAC (Descript Audio Codec) encoder for tokenizing audio, and a transformer decoder
that generates audio tokens conditioned on both the voice description and the text to speak.

## How It Works

**References:**

- Paper: "Natural language guidance of high-fidelity text-to-speech" (Lyth et al., 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParlerTTS(NeuralNetworkArchitecture<>,ParlerTTSOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `ParlerTTS` class in native training/inference mode. |
| `ParlerTTS(NeuralNetworkArchitecture<>,String,ParlerTTSOptions)` | Initializes a new instance of the `ParlerTTS` class in ONNX inference mode. |

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
| `Synthesize(String)` | Synthesizes speech using Parler-TTS's description-guided pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

