---
title: "GeoChat<T>"
description: "GeoChat: grounded VLM for satellite imagery understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.RemoteSensing`

GeoChat: grounded VLM for satellite imagery understanding.

## For Beginners

GeoChat is a vision-language model for grounded understanding
of satellite and aerial imagery. Default values follow the original paper settings.

## How It Works

GeoChat (MBZUAI, 2024) is a grounded large vision-language model specialized for remote
sensing. It provides spatially-aware understanding of satellite and aerial imagery with
capabilities for scene classification, object grounding, region-level captioning, and
visual question answering on remote sensing data.

**References:**

- Paper: "GeoChat: Grounded Large Vision-Language Model for Remote Sensing (MBZUAI, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a satellite image using GeoChat's grounded VLM pipeline. |

