---
title: "AudioSep<T>"
description: "AudioSep - foundation model for open-vocabulary audio separation and sound event detection (Liu et al., ICML 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

AudioSep - foundation model for open-vocabulary audio separation and sound event detection
(Liu et al., ICML 2024).

## For Beginners

AudioSep is like having a smart audio assistant that understands
natural language. You can ask it to find any sound:

- "Find the dog barking" - detects when dogs bark in the audio
- "Extract the piano" - separates piano from other instruments
- "Isolate the speech" - pulls out speech from noisy background

This is powerful because you're not limited to a fixed set of 527 categories - you can
describe any sound in your own words, and AudioSep will find it.

**Usage:**

## How It Works

AudioSep (Liu et al., ICML 2024) is a foundation model that uses natural language queries
to detect and separate sounds. Unlike fixed-vocabulary SED models, AudioSep can detect any
sound described in text using CLAP (Contrastive Language-Audio Pretraining) embeddings to
condition a U-Net separation network. It achieves state-of-the-art results on both sound
separation (MUSDB18, LibriMix) and sound event detection (AudioSet, ESC-50) benchmarks.

**Architecture:**

- **CLAP text encoder**: Encodes the text query (e.g., "dog barking") into a 512-dim

embedding that captures the semantic meaning of the target sound.

- **CLAP audio encoder**: Encodes the input audio mixture into the same embedding space,

enabling audio-text matching.

- **Conditioned U-Net**: A separation network conditioned on the CLAP text embedding.

The text embedding is injected via FiLM (Feature-wise Linear Modulation) at each layer,
telling the network which sound to extract.

- **Classification head**: For SED mode, applies sigmoid activation to produce per-class

probabilities from the audio-text similarity scores.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioSep(NeuralNetworkArchitecture<>,AudioSepOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an AudioSep model for native training mode. |
| `AudioSep(NeuralNetworkArchitecture<>,String,AudioSepOptions)` | Creates an AudioSep model for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` |  |
| `SupportedEvents` |  |
| `TimeResolution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` |  |
| `Detect(Tensor<>,)` |  |
| `DetectAsync(Tensor<>,CancellationToken)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` |  |
| `GetEventProbabilities(Tensor<>)` |  |
| `StartStreamingSession` |  |
| `StartStreamingSession(Int32,)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `AudioSetLabels` | AudioSet-527 standard event labels. |

