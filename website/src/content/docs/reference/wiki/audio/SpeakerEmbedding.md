---
title: "SpeakerEmbedding<T>"
description: "Represents a speaker embedding vector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

Represents a speaker embedding vector.

## For Beginners

SpeakerEmbedding provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `Duration` | Gets or sets the duration of the source audio in seconds. |
| `NumFrames` | Gets or sets the number of frames used. |
| `Vector` | Gets or sets the embedding vector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CosineSimilarity(SpeakerEmbedding<>)` | Computes cosine similarity with another embedding. |
| `EuclideanDistance(SpeakerEmbedding<>)` | Computes Euclidean distance from another embedding. |

