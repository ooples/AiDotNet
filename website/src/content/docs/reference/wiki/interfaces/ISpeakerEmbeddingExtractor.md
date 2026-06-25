---
title: "ISpeakerEmbeddingExtractor<T>"
description: "Interface for speaker embedding extraction models (d-vector/x-vector extraction)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for speaker embedding extraction models (d-vector/x-vector extraction).

## For Beginners

Speaker embeddings are like a "voiceprint" - a compact
representation of what makes someone's voice unique.

How speaker embeddings work:

1. Audio of someone speaking is fed into the model
2. The model outputs a fixed-size vector (e.g., 256 or 512 numbers)
3. This vector captures voice characteristics (pitch, timbre, accent, etc.)
4. Vectors from the same speaker are similar; different speakers are different

Common use cases:

- Voice authentication ("Is this person who they claim to be?")
- Speaker identification ("Who is speaking?")
- Voice cloning (TTS with specific voice)
- Meeting transcription (separating speakers)

Key concepts:

- d-vector: Early embedding approach using DNN
- x-vector: Modern approach using TDNN with statistics pooling
- ECAPA-TDNN: State-of-the-art speaker embedding model

## How It Works

Speaker embedding extractors convert voice audio into fixed-length vectors that
capture the unique characteristics of a speaker's voice. These embeddings enable
speaker verification, identification, and diarization tasks.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimension of output speaker embeddings. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MinimumDurationSeconds` | Gets the minimum audio duration required for reliable embedding extraction. |
| `SampleRate` | Gets the expected sample rate for input audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateEmbeddings(IReadOnlyList<Tensor<>>)` | Aggregates multiple embeddings into a single representative embedding. |
| `ComputeSimilarity(Tensor<>,Tensor<>)` | Computes similarity between two speaker embeddings. |
| `ExtractEmbedding(Tensor<>)` | Extracts speaker embedding from audio. |
| `ExtractEmbeddingAsync(Tensor<>,CancellationToken)` | Extracts speaker embedding from audio asynchronously. |
| `ExtractEmbeddings(IReadOnlyList<Tensor<>>)` | Extracts embeddings from multiple audio segments. |
| `NormalizeEmbedding(Tensor<>)` | Normalizes an embedding for comparison (typically L2 normalization). |

