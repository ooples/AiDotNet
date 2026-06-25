---
title: "Wav2Vec2<T>"
description: "wav2vec 2.0 self-supervised speech representation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Foundations`

wav2vec 2.0 self-supervised speech representation model.

## For Beginners

wav2vec 2.0 was the breakthrough that showed AI could learn to
understand speech with very little labeled data. It works by:

1. Converting raw audio into features with a CNN
2. Masking some features (hiding them)
3. Learning to predict the masked parts from context

This is similar to how GPT predicts the next word, but for audio.

**Usage:**

## How It Works

wav2vec 2.0 (Baevski et al., 2020, Meta) learns speech representations via contrastive
learning over quantized latent speech units. With just 10 minutes of labeled data, it
achieves WER 1.8% on LibriSpeech test-clean when fine-tuned for ASR. It pioneered the
self-supervised approach later extended by HuBERT and WavLM.

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

