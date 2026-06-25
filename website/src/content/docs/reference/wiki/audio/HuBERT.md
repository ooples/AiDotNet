---
title: "HuBERT<T>"
description: "HuBERT (Hidden-Unit BERT) self-supervised speech representation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Foundations`

HuBERT (Hidden-Unit BERT) self-supervised speech representation model.

## For Beginners

HuBERT is a foundation model for audio - it understands speech
at a deep level without needing any labels during training. It can be used as a feature
extractor: feed it audio and get rich embeddings that capture everything about the speech
(what's said, who's speaking, their emotion, etc.).

**Usage:**

## How It Works

HuBERT (Hsu et al., 2021, Meta) learns speech representations by predicting masked
discrete speech units. It uses an offline clustering step to create pseudo-labels from
audio, then trains a BERT-like masked prediction task. HuBERT-Base (12 layers) and
HuBERT-Large (24 layers) achieve state-of-the-art on multiple speech processing tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HuBERT(NeuralNetworkArchitecture<>,HuBERTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a HuBERT model in native training mode. |
| `HuBERT(NeuralNetworkArchitecture<>,String,HuBERTOptions)` | Creates a HuBERT model in ONNX inference mode. |

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

