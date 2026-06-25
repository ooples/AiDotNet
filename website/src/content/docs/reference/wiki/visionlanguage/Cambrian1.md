---
title: "Cambrian1<T>"
description: "Cambrian-1: Spatial Vision Aggregator with 35+ vision encoder combinations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Cambrian-1: Spatial Vision Aggregator with 35+ vision encoder combinations.

## For Beginners

Cambrian-1 takes a unique approach to vision by combining
multiple different vision encoders (CLIP, DINOv2, SigLIP, ConvNeXt — over 35 combinations
tested) rather than using just one. Its Spatial Vision Aggregator (SVA) uses cross-attention
to dynamically select and fuse the most useful features from each encoder based on what part
of the image is relevant. This gives it stronger visual understanding than any single encoder
alone, and it uses LLaMA-3 as its language backbone. Default values follow the original paper
settings.

## How It Works

Cambrian-1 (NYU, 2024) introduces the Spatial Vision Aggregator (SVA) that can combine
features from multiple vision encoders (35+ combinations tested). It uses LLaMA-3 as the
language backbone and demonstrates that diverse vision encoders improve multimodal understanding.

**References:**

- Paper: "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from an image using Cambrian-1's Spatial Vision Aggregator (SVA). |

