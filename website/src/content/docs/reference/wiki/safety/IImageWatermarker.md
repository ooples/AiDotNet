---
title: "IImageWatermarker<T>"
description: "Interface for image watermarking modules that embed and detect watermarks in images."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Watermarking`

Interface for image watermarking modules that embed and detect watermarks in images.

## For Beginners

An image watermarker adds an invisible signature to images.
Even if someone screenshots, crops, or compresses the image, the watermark can
still be detected to prove the image was AI-generated.

## How It Works

Image watermarkers embed imperceptible watermarks in images using frequency domain
(DCT/DWT), neural encoder-decoder, or spatial domain techniques. The watermark
survives common transformations like compression, resizing, and cropping.

**References:**

- SynthID-Image: Internet-scale image watermarking (Google DeepMind, 2025, arxiv:2510.09263)
- Watermarking survey: unified framework (2025, arxiv:2504.03765)

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Tensor<>)` | Detects the watermark confidence score in the given image (0.0 = no watermark, 1.0 = certain). |

