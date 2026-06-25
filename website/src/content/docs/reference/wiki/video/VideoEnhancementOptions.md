---
title: "VideoEnhancementOptions<T>"
description: "Configuration options for video enhancement models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for video enhancement models.

## For Beginners

These options control how your video enhancement model works.
The most important options are:

- ScaleFactor: How much to upscale the video (2x, 4x, etc.)
- EnhancementType: What kind of enhancement to apply
- UseTemporalConsistency: Whether to keep frames looking consistent over time

Example:

## How It Works

Extends `VideoModelOptions` with enhancement-specific settings
like scale factor, temporal smoothing, and loss function configuration.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdversarialLossWeight` | Gets or sets the weight for adversarial (GAN) loss. |
| `EffectiveAdversarialLossWeight` | Gets the effective adversarial loss weight with default fallback. |
| `EffectiveEnhancementType` | Gets the effective enhancement type with default fallback. |
| `EffectiveFlowLossWeight` | Gets the effective flow loss weight with default fallback. |
| `EffectiveL1LossWeight` | Gets the effective L1 loss weight with default fallback. |
| `EffectiveNumFeatureChannels` | Gets the effective number of feature channels with default fallback. |
| `EffectiveNumResidualBlocks` | Gets the effective number of residual blocks with default fallback. |
| `EffectivePerceptualLossWeight` | Gets the effective perceptual loss weight with default fallback. |
| `EffectiveRecurrentIterations` | Gets the effective recurrent iterations with default fallback. |
| `EffectiveScaleFactor` | Gets the effective scale factor with default fallback. |
| `EffectiveTemporalLossWeight` | Gets the effective temporal loss weight with default fallback. |
| `EffectiveTemporalScaleFactor` | Gets the effective temporal scale factor with default fallback. |
| `EffectiveUseAttention` | Gets the effective attention setting with default fallback. |
| `EffectiveUseBidirectional` | Gets the effective bidirectional setting with default fallback. |
| `EffectiveUseTemporalConsistency` | Gets the effective temporal consistency setting with default fallback. |
| `EnhancementType` | Gets or sets the type of enhancement to perform. |
| `FlowLossWeight` | Gets or sets the weight for flow-warping loss. |
| `L1LossWeight` | Gets or sets the weight for pixel-wise L1 loss. |
| `NumFeatureChannels` | Gets or sets the number of feature channels in the model. |
| `NumResidualBlocks` | Gets or sets the number of residual blocks in the model. |
| `PerceptualLossWeight` | Gets or sets the weight for perceptual loss (VGG-based). |
| `RecurrentIterations` | Gets or sets the number of recurrent iterations for temporal models. |
| `ScaleFactor` | Gets or sets the spatial scale factor for upscaling. |
| `TemporalLossWeight` | Gets or sets the weight for temporal consistency loss. |
| `TemporalScaleFactor` | Gets or sets the temporal scale factor for frame rate increase. |
| `UseAttention` | Gets or sets whether to use attention mechanisms in residual blocks. |
| `UseBidirectional` | Gets or sets whether to use bidirectional propagation for temporal models. |
| `UseTemporalConsistency` | Gets or sets whether to enforce temporal consistency between frames. |

