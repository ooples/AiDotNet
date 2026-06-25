---
title: "ControlNetInpaintingModel<T>"
description: "ControlNet Inpainting model with mask-aware conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet Inpainting model with mask-aware conditioning.

## For Beginners

This model erases part of an image (marked by a mask) and
fills it in with new content that matches a control signal. For example, you could
erase a person from a photo and use an edge map to guide what replaces them.

## How It Works

Specialized ControlNet for inpainting that takes both a control image and a binary
mask as input. The mask indicates which regions to regenerate while the control
signal guides the structure of the inpainted content.

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

