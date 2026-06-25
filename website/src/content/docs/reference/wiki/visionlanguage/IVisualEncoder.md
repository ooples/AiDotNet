---
title: "IVisualEncoder<T>"
description: "Interface for vision encoders that extract feature representations from images."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for vision encoders that extract feature representations from images.

## For Beginners

A visual encoder is like a "feature extractor" for images. It reads an image
and produces a list of numbers (an embedding) that summarizes what the image contains. Two similar
images will have similar embeddings, making it easy for other models to compare and reason about images.

## How It Works

Visual encoders transform raw image data (pixel tensors) into compact feature representations
(embeddings) that capture semantic content. These embeddings can be used for:

- Image classification (zero-shot or fine-tuned)
- Image-text similarity search and retrieval
- Visual question answering (as input to downstream VLMs)
- Object detection and segmentation (as backbone features)

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimensionality of the output embedding space. |
| `ImageChannels` | Gets the number of image channels expected (typically 3 for RGB). |
| `ImageSize` | Gets the expected input image size (height and width in pixels). |

## Methods

| Method | Summary |
|:-----|:--------|
| `EncodeImage(Tensor<>)` | Extracts a visual embedding from an image tensor. |

