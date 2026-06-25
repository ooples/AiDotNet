---
title: "AudioTextDualStreamArchitecture<T>"
description: "A neural network architecture for audio + text two-stream models (CLAP-family encoders) that hosts a separate audio encoder and text encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

A neural network architecture for audio + text two-stream models
(CLAP-family encoders) that hosts a separate audio encoder and text encoder.

## How It Works

Concrete audio-language specialisation of `DualEncoderArchitecture`.
The abstract base owns the two encoder stacks under modality-neutral names
(`EncoderALayers` / `EncoderBLayers`); this subclass exposes them
under their semantic aliases `AudioLayers` / `TextLayers` so
CLAP-family code reads naturally.

CLAP (Wu et al. 2023) and similar audio-language pretraining models train
two parallel encoders: an audio side (HTSAT / PANNs / Wav2Vec / etc.) and a
text side (RoBERTa / BERT / etc.). They only meet at the contrastive
objective, so each side gets its own customisable layer stack.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioTextDualStreamArchitecture(IEnumerable<ILayer<>>,IEnumerable<ILayer<>>,InputType,NeuralNetworkTaskType,NetworkComplexity,Int32,Int32,Int32,Int32,Boolean)` | Initializes a new audio-text dual-stream architecture with explicit audio and text encoder layer stacks. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioLayers` | Gets the layer stack for the audio encoder. |
| `TextLayers` | Gets the layer stack for the text encoder. |

