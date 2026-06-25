---
title: "IMultimodalEmbedding<T>"
description: "Interface for multimodal embedding models that can encode multiple modalities (text, images, audio) into a shared embedding space."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for multimodal embedding models that can encode multiple modalities
(text, images, audio) into a shared embedding space.

## For Beginners

Imagine you want to search for images using text queries.
A multimodal model learns to convert both "a photo of a cat" and an actual cat image
into similar vectors, allowing direct comparison between text and images.

## How It Works

Multimodal embedding models like CLIP (Contrastive Language-Image Pre-training)
learn to project different types of data into the same vector space, enabling
cross-modal similarity search and zero-shot classification.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimensionality of the embedding space. |
| `ImageSize` | Gets the expected image size (square images: ImageSize x ImageSize pixels). |
| `MaxSequenceLength` | Gets the maximum sequence length for text input. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSimilarity(Vector<>,Vector<>)` | Computes similarity between two embeddings. |
| `EncodeImage(Double[])` | Encodes an image into an embedding vector. |
| `EncodeImageBatch(IEnumerable<Double[]>)` | Encodes multiple images into embedding vectors in a batch. |
| `EncodeText(String)` | Encodes text into an embedding vector. |
| `EncodeTextBatch(IEnumerable<String>)` | Encodes multiple texts into embedding vectors in a batch. |
| `ZeroShotClassify(Double[],IEnumerable<String>)` | Performs zero-shot classification of an image against text labels. |

