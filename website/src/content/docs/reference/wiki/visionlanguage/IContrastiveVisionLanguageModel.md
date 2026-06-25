---
title: "IContrastiveVisionLanguageModel<T>"
description: "Interface for contrastive vision-language models that align image and text embeddings in a shared space for zero-shot classification and cross-modal retrieval."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for contrastive vision-language models that align image and text embeddings
in a shared space for zero-shot classification and cross-modal retrieval.

## For Beginners

Think of this like a universal translator between images and text.
The model learns that a photo of a dog and the text "a photo of a dog" should have similar
representations. This means you can search for images using text, or classify images by
comparing them with text descriptions - without training on specific categories.

## How It Works

Contrastive VLMs (like CLIP, SigLIP, ALIGN) learn to place matching image-text pairs close
together in a shared embedding space while pushing non-matching pairs apart. This enables:

- **Zero-shot classification**: Classify images using text descriptions without training
- **Image-text retrieval**: Find images matching a text query, or texts describing an image
- **Similarity scoring**: Measure how well an image matches a text description

## Properties

| Property | Summary |
|:-----|:--------|
| `ProjectionDimension` | Gets the dimensionality of the shared projection space. |
| `Temperature` | Gets the temperature parameter used for similarity scaling. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSimilarity(Tensor<>,String)` | Computes the similarity score between an image and a text description. |
| `ZeroShotClassify(Tensor<>,String[])` | Performs zero-shot image classification by comparing the image with text labels. |

