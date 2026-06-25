---
title: "MeloTTS<T>"
description: "MeloTTS: high-quality multilingual TTS with VITS backbone, language-specific text processing, and mixed-language support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.EndToEnd`

MeloTTS: high-quality multilingual TTS with VITS backbone, language-specific text processing, and mixed-language support.

## For Beginners

MeloTTS builds on the VITS architecture (see VITS class) to add robust
multilingual support. While VITS handles the core speech synthesis pipeline (VAE + normalizing flows +
HiFi-GAN decoder), MeloTTS adds several important features:
(1) Language-specific text processing using BERT-based grapheme-to-phoneme (G2P) for Chinese
and eSpeak for other languages,
(2) Language ID embeddings that tell the encoder and decoder which language is being spoken,
(3) Mixed-language (code-switching) support so it can handle sentences that switch between
languages mid-utterance, and
(4) Multi-speaker conditioning for generating different voices.
This makes it particularly useful for applications needing natural speech across multiple languages.

## How It Works

**References:**

- Project: "MeloTTS: High-quality Multi-lingual Text-to-Speech" (MyShell, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeloTTS(NeuralNetworkArchitecture<>,MeloTTSOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the `MeloTTS` class in native training/inference mode. |
| `MeloTTS(NeuralNetworkArchitecture<>,String,MeloTTSOptions)` | Initializes a new instance of the `MeloTTS` class in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimension size. |
| `NumFlowSteps` | Gets the number of normalizing flow steps used in the VITS backbone. |

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
| `Synthesize(String)` | Synthesizes multilingual speech using MeloTTS's extended VITS pipeline. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

