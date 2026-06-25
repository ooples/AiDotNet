---
title: "ClipNeuralNetwork<T>"
description: "CLIP (Contrastive Language-Image Pre-training) neural network that encodes both text and images into a shared embedding space, enabling cross-modal similarity and zero-shot classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

CLIP (Contrastive Language-Image Pre-training) neural network that encodes both text
and images into a shared embedding space, enabling cross-modal similarity and zero-shot classification.

## For Beginners

CLIP learns to connect images and text by training on millions
of image-caption pairs from the internet. It creates a shared space where similar images
and text descriptions are close together. This enables powerful capabilities like zero-shot
image classification (categorizing images without specific training), image search using
text queries, and measuring how well an image matches a description.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClipNeuralNetwork(NeuralNetworkArchitecture<>,String,String,ITokenizer,ILossFunction<>,Int32,Int32,Int32,ClipOptions)` | Initializes a new instance of the CLIP neural network. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension of the CLIP model. |
| `ImageSize` | Gets the expected image size (square images: ImageSize x ImageSize pixels). |
| `MaxSequenceLength` | Gets the maximum sequence length for text input. |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `EncodeText(String)` |  |
| `EncodeTextBatch(IEnumerable<String>)` |  |
| `GetOptions` |  |

