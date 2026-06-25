---
title: "IBlipModel<T>"
description: "Defines the contract for BLIP (Bootstrapped Language-Image Pre-training) models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for BLIP (Bootstrapped Language-Image Pre-training) models.

## For Beginners

BLIP is like CLIP but with extra superpowers!

What CLIP can do:

- Compare images and text (are they related?)
- Zero-shot classification (classify without training)

What BLIP adds:

- Generate captions for images (describe what you see)
- Answer questions about images (VQA)
- Better image-text matching with cross-attention

BLIP was trained on a larger, cleaner dataset using a special "bootstrapping"
technique that improves the quality of training data automatically.

## How It Works

BLIP extends CLIP's capabilities with additional vision-language tasks:
image captioning, image-text matching, and visual question answering.
This interface extends `IMultimodalEmbedding` with these features.

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String,Int32)` | Answers a question about an image's content. |
| `ComputeImageTextMatch(Tensor<>,String)` | Determines whether a given text accurately describes an image. |
| `GenerateCaption(Tensor<>,Int32,Int32)` | Generates a caption describing the content of an image. |
| `GenerateCaptions(Tensor<>,Int32,Int32)` | Generates multiple candidate captions for an image. |
| `RankCaptions(Tensor<>,IEnumerable<String>)` | Ranks a set of candidate captions by how well they match an image. |
| `RetrieveImages(String,IEnumerable<Vector<>>,Int32)` | Retrieves the most relevant images for a text query from a collection. |
| `RetrieveTexts(Tensor<>,IEnumerable<Vector<>>,Int32)` | Retrieves the most relevant texts for an image from a collection. |

