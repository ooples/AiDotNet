---
title: "SpeakerLMOptions"
description: "Configuration options for the SpeakerLM model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for the SpeakerLM model.

## For Beginners

SpeakerLM treats speaker recognition like a language problem - it
"reads" speaker characteristics the way a language model reads words. This lets it handle
complex multi-speaker scenarios where multiple people are talking.

## How It Works

SpeakerLM (2024) uses a language model backbone for speaker understanding tasks including
speaker verification, diarization, and speaker-attributed transcription. It processes
speaker embeddings as tokens in a sequence, enabling multi-speaker understanding in a
unified framework.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultThreshold` | Gets or sets the default verification threshold (cosine similarity). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the speaker embedding dimension. |
| `LMHiddenDim` | Gets or sets the language model hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxSpeakers` | Gets or sets the maximum number of speakers. |
| `MinDurationSeconds` | Gets or sets the minimum audio duration in seconds for reliable embedding extraction. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLMLayers` | Gets or sets the number of LM layers. |
| `NumMels` | Gets or sets the number of mel-frequency bins for audio preprocessing. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant. |

