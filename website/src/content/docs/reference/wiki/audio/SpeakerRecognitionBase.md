---
title: "SpeakerRecognitionBase<T>"
description: "Base class for speaker recognition models (embedding extraction, verification, diarization)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.Speaker`

Base class for speaker recognition models (embedding extraction, verification, diarization).

## For Beginners

Speaker recognition is like voice fingerprinting.
Just as fingerprints are unique to each person, voice characteristics (pitch,
speaking style, accent) can identify individuals.

This base class provides:

- Feature extraction utilities (MFCCs, spectral features)
- Embedding dimension management
- Similarity computation methods

## How It Works

Speaker recognition encompasses tasks that identify or verify speakers based on their voice.
This base class provides common functionality for:

- Speaker embedding extraction (d-vectors, x-vectors)
- Speaker verification (is this the claimed speaker?)
- Speaker diarization (who spoke when?)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeakerRecognitionBase(NeuralNetworkArchitecture<>,ILossFunction<>)` | Initializes a new instance of the SpeakerRecognitionBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimension of output speaker embeddings. |
| `MfccExtractor` | Gets the MFCC extractor for preprocessing. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateEmbeddings(IReadOnlyList<Tensor<>>)` | Aggregates multiple embeddings into a single representative embedding. |
| `ComputeCosineSimilarity(Tensor<>,Tensor<>)` | Computes cosine similarity between two speaker embedding tensors. |
| `ComputeCosineSimilarity(Vector<>,Vector<>)` | Computes cosine similarity between two speaker embeddings. |
| `CreateMfccExtractor(Int32,Int32)` | Creates an MFCC extractor for preprocessing speaker audio. |
| `NormalizeEmbedding(Tensor<>)` | Normalizes an embedding to unit length (L2 normalization). |

