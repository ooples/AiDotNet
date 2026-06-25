---
title: "Tacotron2<T>"
description: "Tacotron 2: improved attention-based TTS with location-sensitive attention and simplified decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Classic`

Tacotron 2: improved attention-based TTS with location-sensitive attention and simplified decoder.

## For Beginners

Tacotron 2 is an improved version of Tacotron that produces higher-quality speech.
It generates mel-spectrograms from text using an encoder-decoder architecture with attention,
then a separate vocoder (like WaveNet) converts the mel-spectrogram into an audio waveform.

## How It Works

**References:**

- Paper: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Shen et al., 2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Tacotron2(NeuralNetworkArchitecture<>,String,Tacotron2Options)` | Initializes a new instance of the `Tacotron2` class in ONNX inference mode. |
| `Tacotron2(NeuralNetworkArchitecture<>,Tacotron2Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `Tacotron2` class in native training/inference mode. |

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
| `Synthesize(String)` | Synthesizes mel-spectrogram from text using Tacotron 2's pipeline. |
| `TextToMel(String)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

