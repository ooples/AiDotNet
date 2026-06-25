---
title: "BridgeTower<T>"
description: "BridgeTower: cross-modal alignment through bridge layers connecting vision and text encoder layers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

BridgeTower: cross-modal alignment through bridge layers connecting vision and text encoder layers.

## For Beginners

BridgeTower is a vision-language model. Default values follow the original paper settings.

## How It Works

BridgeTower (Xu et al., AAAI 2023) introduces bridge layers that connect vision and text encoder
layers at multiple levels, enabling fine-grained cross-modal alignment. Each bridge layer consists
of cross-attention between corresponding encoder layers, creating bidirectional information flow
throughout the encoding process.

**References:**

- Paper: "BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning" (Xu et al., AAAI 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

