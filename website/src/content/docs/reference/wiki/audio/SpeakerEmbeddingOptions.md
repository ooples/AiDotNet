---
title: "SpeakerEmbeddingOptions"
description: "Configuration options for speaker embedding extraction."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for speaker embedding extraction.

## For Beginners

These options configure the SpeakerEmbedding model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeakerEmbeddingOptions` | Initializes a new instance with default values. |
| `SpeakerEmbeddingOptions(SpeakerEmbeddingOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets or sets the embedding dimension. |
| `FftSize` | Gets or sets the FFT size. |
| `HopLength` | Gets or sets the hop length. |
| `ModelPath` | Gets or sets the path to the neural embedding model. |
| `NumMfcc` | Gets or sets the number of MFCC coefficients. |
| `OnnxOptions` | Gets or sets the ONNX options. |
| `SampleRate` | Gets or sets the sample rate. |

