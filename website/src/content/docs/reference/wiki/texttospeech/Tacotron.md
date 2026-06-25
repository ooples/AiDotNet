---
title: "Tacotron<T>"
description: "Tacotron: sequence-to-sequence attention-based TTS with CBHG encoder and autoregressive decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Classic`

Tacotron: sequence-to-sequence attention-based TTS with CBHG encoder and autoregressive decoder.

## For Beginners

Tacotron is an attention-based text-to-speech model that converts text input into speech audio output.
It uses a sequence-to-sequence architecture where an encoder processes text characters and a decoder generates
mel-spectrogram frames one at a time (autoregressively), using attention to align text with audio.

## How It Works

**References:**

- Paper: "Tacotron: Towards End-to-End Speech Synthesis" (Wang et al., 2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Tacotron(NeuralNetworkArchitecture<>,String,TacotronOptions)` | Initializes a new instance of the `Tacotron` class in ONNX inference mode. |
| `Tacotron(NeuralNetworkArchitecture<>,TacotronOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `Tacotron` class in native training/inference mode. |

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
| `Synthesize(String)` | Synthesizes mel-spectrogram from text using Tacotron's autoregressive pipeline. |
| `TextToMel(String)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

