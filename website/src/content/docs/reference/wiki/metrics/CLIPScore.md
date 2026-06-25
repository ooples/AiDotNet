---
title: "CLIPScore<T>"
description: "CLIPScore metric for evaluating text-image alignment and image quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

CLIPScore metric for evaluating text-image alignment and image quality.

## How It Works

CLIPScore measures how well an image matches a text description (or reference image)
using CLIP embeddings. It's widely used for evaluating text-to-image generation models.

Two main variants:

- CLIPScore (text-image): Measures alignment between generated images and text prompts
- RefCLIPScore (image-image): Measures similarity between generated and reference images

Typical CLIPScore values (0-100 scale):

- >30: Excellent alignment (image matches text well)
- 25-30: Good alignment
- 20-25: Moderate alignment
- <20: Poor alignment

Based on "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
by Hessel et al. (2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLIPScore(IMultimodalEmbedding<>)` | Initializes a new instance of CLIPScore calculator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClipModel` | Gets the CLIP model used for computing embeddings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCaptionScore(Double[],String,IEnumerable<String>)` | Computes CLIPScore for image captioning evaluation. |
| `ComputeCombinedScore(Double[],String,Double[],Double)` | Computes combined CLIPScore using both text-image and image-image similarity. |
| `ComputeDirectionalSimilarity(Double[],Double[],String,String)` | Computes directional similarity for image editing evaluation. |
| `ComputeImageImageScore(Double[],Double[])` | Computes RefCLIPScore between a generated image and a reference image. |
| `ComputeScoreImprovement(Double[],Double[],String)` | Computes CLIPScore improvement between before and after images. |
| `ComputeTextImageScore(Double[],String)` | Computes CLIPScore between an image and a text description. |
| `ComputeTextImageScoreBatch(IList<Double[]>,IList<String>)` | Computes CLIPScore for a batch of images and their corresponding text descriptions. |
| `SubtractVectors(Vector<>,Vector<>)` | Subtracts one vector from another. |

