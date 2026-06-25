---
title: "IAudioFoundationModel<T>"
description: "Defines the contract for self-supervised audio foundation models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for self-supervised audio foundation models.

## For Beginners

A foundation model is like a student who has read millions of
books but hasn't been told what to look for. It develops a deep understanding of language
(or in this case, audio). When you need it for a specific task (like recognizing emotions),
you just teach it the final step - it already understands the audio.

These models provide embeddings (numerical representations) that capture:

- Phonetic content (what sounds are being made)
- Speaker characteristics (who is speaking)
- Prosody and emotion (how they're speaking)
- Acoustic environment (where the recording was made)

Examples: HuBERT, wav2vec 2.0, WavLM, data2vec

## How It Works

Audio foundation models learn general-purpose audio representations through self-supervised
pre-training on large unlabeled datasets. These representations can be fine-tuned for
downstream tasks like speech recognition, speaker verification, and emotion detection.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension of the model. |
| `NumLayers` | Gets the number of transformer layers in the model. |
| `SampleRate` | Gets the sample rate this model operates at. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractEmbeddings(Tensor<>)` | Extracts the final-layer embeddings from audio. |
| `ExtractEmbeddingsAsync(Tensor<>,CancellationToken)` | Extracts embeddings asynchronously. |
| `ExtractLayerFeatures(Tensor<>,Int32)` | Extracts features from a specific layer. |
| `ExtractWeightedFeatures(Tensor<>,[])` | Extracts weighted combination of features from all layers. |

