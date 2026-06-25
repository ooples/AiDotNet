---
title: "Chameleon<T>"
description: "Chameleon: early fusion with discrete tokens for all modalities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

Chameleon: early fusion with discrete tokens for all modalities.

## For Beginners

Chameleon is a unified model that handles text and images
using the same discrete token vocabulary. Default values follow the original paper
settings.

## How It Works

Chameleon (Meta, 2024) is a mixed-modal early-fusion foundation model that represents all
modalities (text, images, code) as discrete tokens in a unified vocabulary. Images are
quantized via a VQ-VAE encoder into discrete visual tokens, enabling a single autoregressive
transformer to generate and understand any combination of text and images natively.

**References:**

- Paper: "Chameleon: Mixed-Modal Early-Fusion Foundation Models" (Meta, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from an image using Chameleon's early-fusion mixed-modal approach. |
| `GenerateImage(String)` | Generates an image from text using Chameleon's early-fusion discrete token generation. |

