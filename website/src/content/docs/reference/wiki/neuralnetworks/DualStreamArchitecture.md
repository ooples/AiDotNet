---
title: "DualStreamArchitecture<T>"
description: "A neural network architecture for vision + text two-stream models (CLIP-family encoders: BASIC, DFNCLIP, EVACLIP, LiT, RegionCLIP, etc.) that hosts a separate vision encoder and text encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

A neural network architecture for vision + text two-stream models
(CLIP-family encoders: BASIC, DFNCLIP, EVACLIP, LiT, RegionCLIP, etc.) that
hosts a separate vision encoder and text encoder.

## For Beginners

Some vision-language models have two separate
"brains" — one that looks at images and one that reads text — and they only
meet at the very end when comparing embeddings. If you want to swap in a
custom layer stack for both halves, this architecture lets you describe
both halves at once, instead of cramming them into a single flat list
(which silently drops the text half).

## How It Works

Concrete vision-language specialisation of `DualEncoderArchitecture`:
the abstract base holds the two encoder stacks under modality-neutral names
(`EncoderALayers` / `EncoderBLayers`); this subclass exposes them
under their semantic vision-language aliases `VisionLayers` /
`TextLayers` so call sites read naturally.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DualStreamArchitecture(IEnumerable<ILayer<>>,IEnumerable<ILayer<>>,InputType,NeuralNetworkTaskType,NetworkComplexity,Int32,Int32,Int32,Int32,Int32,Boolean,Int32,Int32,Int32)` | Initializes a new dual-stream architecture with explicit vision and text encoder layers. |

## Properties

| Property | Summary |
|:-----|:--------|
| `TextLayers` | Gets the layer stack for the text encoder. |
| `VisionLayers` | Gets the layer stack for the vision encoder. |

