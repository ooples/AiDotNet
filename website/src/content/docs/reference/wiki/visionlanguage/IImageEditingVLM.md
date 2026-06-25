---
title: "IImageEditingVLM<T>"
description: "Interface for VLMs that edit images based on natural language instructions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for VLMs that edit images based on natural language instructions.

## How It Works

Image editing VLMs take an input image and a text instruction and produce an edited version.
These models understand the semantic intent of the instruction and apply targeted modifications.

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputImageSize` | Gets the output image resolution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EditImage(Tensor<>,String)` | Edits an image according to a natural language instruction. |

