---
title: "NeuralImageWatermarker<T>"
description: "Image watermarker that uses an encoder-decoder neural network approach for embedding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Image watermarker that uses an encoder-decoder neural network approach for embedding.

## For Beginners

This watermarker uses AI-based techniques to embed signatures
that are extremely hard to remove. The watermark is woven into the image at a deep
level that survives even aggressive editing operations.

## How It Works

Simulates neural watermarking by analyzing learned feature patterns. Neural watermarks
encode information in the latent space of an encoder-decoder network, making them
robust to a wide range of image transformations including cropping, rotation, and
color adjustment.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralImageWatermarker(Double)` | Initializes a new neural image watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Tensor<>)` |  |
| `EvaluateImage(Tensor<>)` |  |

