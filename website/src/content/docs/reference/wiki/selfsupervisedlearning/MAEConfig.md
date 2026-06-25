---
title: "MAEConfig"
description: "MAE-specific configuration settings."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SelfSupervisedLearning`

MAE-specific configuration settings.

## For Beginners

MAE (Masked Autoencoder) learns by masking random patches
of an image and training the model to reconstruct them.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderDepth` | Gets or sets the number of decoder transformer blocks. |
| `DecoderEmbedDimension` | Gets or sets the decoder embedding dimension. |
| `DecoderNumHeads` | Gets or sets the number of decoder attention heads. |
| `MaskRatio` | Gets or sets the fraction of patches to mask. |
| `NormalizeTarget` | Gets or sets whether to normalize reconstruction target. |
| `PatchSize` | Gets or sets the patch size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

