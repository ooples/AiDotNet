---
title: "MERTOptions"
description: "Configuration options for the MERT music understanding foundation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Foundations`

Configuration options for the MERT music understanding foundation model.

## For Beginners

MERT is like HuBERT but specialized for music instead of speech.
While HuBERT learns by predicting speech units, MERT learns by predicting musical features
like pitch and harmony. This means it deeply understands music structure and can be used
for tasks like genre classification, instrument detection, and music tagging.

## How It Works

MERT (Li et al., 2024) is a self-supervised music understanding model that uses acoustic
and musical tokenizers to learn rich music representations. Unlike speech-focused models,
MERT incorporates music-specific knowledge through CQT-based teacher targets and codebook
clustering, enabling strong performance on 14 music information retrieval tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `CQTBins` | Gets or sets the number of CQT bins for music teacher targets. |
| `CodebookSize` | Gets or sets the codebook vocabulary size. |
| `ConvChannels` | Gets or sets the convolutional feature extractor channels. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FeedForwardDim` | Gets or sets the feed-forward inner dimension. |
| `HiddenDim` | Gets or sets the model hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaskProbability` | Gets or sets the masking probability. |
| `MaskSpanLength` | Gets or sets the mask span length. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumClusters` | Gets or sets the number of K-means clusters for target quantization. |
| `NumCodebooks` | Gets or sets the number of RVQ codebooks for acoustic teacher. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("base" with 95M params or "large" with 330M). |

