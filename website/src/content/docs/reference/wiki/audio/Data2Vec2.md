---
title: "Data2Vec2<T>"
description: "data2vec 2.0 self-supervised audio representation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Foundations`

data2vec 2.0 self-supervised audio representation model.

## For Beginners

data2vec 2.0 learns audio features by predicting its own hidden
representations - like learning by explaining things to yourself. Unlike HuBERT which
predicts discrete labels, data2vec predicts rich continuous features. It can be used as a
powerful feature extractor for any downstream audio task.

**Usage:**

## How It Works

data2vec 2.0 (Baevski et al., 2023, Meta) is a self-supervised framework that predicts
contextualized latent representations rather than modality-specific targets. Version 2.0 is
16x faster than v1 through efficient data encoding and self-distillation. It achieves strong
results on speech, vision, and language tasks with the same method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Data2Vec2(NeuralNetworkArchitecture<>,Data2Vec2Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a data2vec 2.0 model in native training mode. |
| `Data2Vec2(NeuralNetworkArchitecture<>,String,Data2Vec2Options)` | Creates a data2vec 2.0 model in ONNX inference mode. |

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

