---
title: "RegionCLIP<T>"
description: "RegionCLIP model extending CLIP to learn region-level (object-level) visual representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

RegionCLIP model extending CLIP to learn region-level (object-level) visual representations.

## For Beginners

RegionCLIP extends CLIP from whole-image understanding to
individual object regions within images. While CLIP matches an entire image to text,
RegionCLIP learns to match specific regions (bounding boxes) to text descriptions,
enabling zero-shot object detection — finding objects in images by describing them in
natural language. Default values follow the original paper settings.

## How It Works

RegionCLIP (Zhong et al., CVPR 2022) generates region-text pairs from image captions using object
proposals and learns to align individual image regions with text descriptions, enabling zero-shot
and open-vocabulary object detection.

**References:**

- Paper: "RegionCLIP: Region-based Language-Image Pretraining" (Zhong et al., CVPR 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

