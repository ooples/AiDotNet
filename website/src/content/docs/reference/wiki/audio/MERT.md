---
title: "MERT<T>"
description: "MERT self-supervised music understanding foundation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Foundations`

MERT self-supervised music understanding foundation model.

## For Beginners

MERT is like HuBERT but for music. It deeply understands musical
structure - pitch, harmony, rhythm, instrumentation - without needing labeled data. Use it
as a feature extractor: feed it music and get embeddings useful for genre classification,
instrument detection, mood analysis, and more.

**Usage:**

## How It Works

MERT (Li et al., 2024) uses acoustic and musical tokenizers to learn rich music
representations. Unlike speech models, it incorporates music-specific knowledge through
CQT-based teacher targets and codebook clustering, enabling strong performance on 14
music information retrieval tasks including tagging, genre, instrument, and key detection.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MERT(NeuralNetworkArchitecture<>,MERTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a MERT model in native training mode. |
| `MERT(NeuralNetworkArchitecture<>,String,MERTOptions)` | Creates a MERT model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `NumLayers` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractEmbeddings(Tensor<>)` |  |
| `ExtractEmbeddingsAsync(Tensor<>,CancellationToken)` |  |
| `ExtractLayerFeatures(Tensor<>,Int32)` |  |
| `ExtractWeightedFeatures(Tensor<>,[])` |  |
| `PostprocessOutput(Tensor<>)` | Returns output unchanged; no post-processing needed for foundation model embeddings. |
| `PreprocessAudio(Tensor<>)` | Returns raw audio unchanged; MERT expects raw waveform input and handles internal feature extraction. |

