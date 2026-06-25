---
title: "MusicFlamingoOptions"
description: "Configuration options for the Music Flamingo model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Multimodal`

Configuration options for the Music Flamingo model.

## For Beginners

Music Flamingo is like giving a language AI the ability to understand music.
You can play it a song and ask "What genre is this?" or "What instruments are playing?" and it
will answer in natural language, combining its music listening ability with language understanding.

## How It Works

Music Flamingo (2024) adapts the Flamingo architecture specifically for music understanding.
It uses a frozen music encoder (e.g., MERT or Jukebox features) with perceiver cross-attention
to enable a pre-trained LLM to reason about music: answering questions about genre, instruments,
mood, structure, and musical theory.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LLMHiddenDim` | Gets or sets the LLM hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxAudioDurationSeconds` | Gets or sets the maximum audio duration in seconds. |
| `MaxResponseTokens` | Gets or sets the maximum response tokens. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `MusicEncoderDim` | Gets or sets the music encoder dimension. |
| `NumPerceiverLayers` | Gets or sets the number of perceiver layers. |
| `NumPerceiverTokens` | Gets or sets the number of perceiver latent tokens. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant. |

