---
title: "VideoModelOptions<T>"
description: "Base configuration options for all video AI models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Base configuration options for all video AI models.

## For Beginners

You don't need to set any of these options to get started!
The model will use sensible defaults for everything. Only change options if you
know you need something different.

Example - Using defaults:

Example - Customizing some options:

## How It Works

All options use nullable properties with industry-standard defaults applied internally.
This allows zero-configuration usage while still enabling full customization.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EffectiveDropoutRate` | Gets the effective dropout rate with default fallback. |
| `EffectiveHiddenDimension` | Gets the effective hidden dimension with default fallback. |
| `EffectiveInputChannels` | Gets the effective number of input channels with default fallback. |
| `EffectiveInputHeight` | Gets the effective input height with default fallback. |
| `EffectiveInputWidth` | Gets the effective input width with default fallback. |
| `EffectiveLearningRate` | Gets the effective learning rate with default fallback. |
| `EffectiveMaxGradientNorm` | Gets the effective max gradient norm with default fallback. |
| `EffectiveNumAttentionHeads` | Gets the effective number of attention heads with default fallback. |
| `EffectiveNumFrames` | Gets the effective number of frames with default fallback. |
| `EffectiveNumLayers` | Gets the effective number of layers with default fallback. |
| `EffectiveUseGradientClipping` | Gets the effective gradient clipping setting with default fallback. |
| `EffectiveWeightDecay` | Gets the effective weight decay with default fallback. |
| `HiddenDimension` | Gets or sets the hidden dimension size for the model. |
| `InputChannels` | Gets or sets the number of input channels. |
| `InputHeight` | Gets or sets the expected input height in pixels. |
| `InputWidth` | Gets or sets the expected input width in pixels. |
| `LearningRate` | Gets or sets the learning rate for optimization. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for clipping. |
| `NumAttentionHeads` | Gets or sets the number of attention heads for transformer-based models. |
| `NumFrames` | Gets or sets the expected number of input frames. |
| `NumLayers` | Gets or sets the number of transformer/encoder layers. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `UseGpu` | Gets or sets whether to use GPU acceleration if available. |
| `UseGradientClipping` | Gets or sets whether to use gradient clipping. |
| `UseMixedPrecision` | Gets or sets whether to use mixed precision (FP16) for faster computation. |
| `WeightDecay` | Gets or sets the weight decay for L2 regularization. |

