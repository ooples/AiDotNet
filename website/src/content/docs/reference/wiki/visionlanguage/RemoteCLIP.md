---
title: "RemoteCLIP<T>"
description: "RemoteCLIP model adapted for remote sensing and satellite imagery understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

RemoteCLIP model adapted for remote sensing and satellite imagery understanding.

## For Beginners

RemoteCLIP adapts CLIP for satellite and aerial imagery by
fine-tuning on curated remote sensing image-text datasets. It enables zero-shot scene
classification of satellite images (e.g., "airport", "farmland"), image-text retrieval,
and object counting in aerial views without task-specific training. Default values follow
the original paper settings.

## How It Works

RemoteCLIP (Liu et al., 2023) fine-tunes CLIP on curated remote sensing image-text datasets
for zero-shot scene classification, image-text retrieval, and object counting in satellite
and aerial imagery.

**References:**

- Paper: "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing" (Liu et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

