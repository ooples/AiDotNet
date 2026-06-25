---
title: "VisionTSOptions<T>"
description: "Configuration options for VisionTS (Visual Masked Autoencoders as Zero-Shot Time Series Forecasters)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for VisionTS (Visual Masked Autoencoders as Zero-Shot Time Series Forecasters).

## For Beginners

VisionTS brings image AI to time series:

**Cross-Modal Transfer:**
VisionTS converts time series into 2D image-like representations, then uses
a pretrained Visual MAE (originally trained on images) to process them. This
leverages the massive pretraining of vision models for time series tasks.

**How It Works:**

1. Convert time series to 2D patch grid (like an image)
2. Mask some patches (MAE-style)
3. Use the pretrained ViT encoder to process visible patches
4. Decode masked patches to reconstruct/forecast the series

## How It Works

VisionTS repurposes Visual Masked Autoencoders (MAE) for time series forecasting,
demonstrating that vision foundation models can transfer effectively to the time
series domain through cross-modal transfer.

**Reference:** "VisionTS: Visual Masked Autoencoders as Zero-Shot Time Series Forecasters",
ICML 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisionTSOptions` | Initializes a new instance with default values. |
| `VisionTSOptions(VisionTSOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension of the ViT encoder. |
| `ImageHeight` | Gets or sets the image height for 2D conversion. |
| `ImageWidth` | Gets or sets the image width for 2D conversion. |
| `IntermediateSize` | Gets or sets the intermediate size. |
| `MaskRatio` | Gets or sets the mask ratio for MAE pretraining. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `PatchLength` | Gets or sets the patch length for 2D patch grid conversion. |

