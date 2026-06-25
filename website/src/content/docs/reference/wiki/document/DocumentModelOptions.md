---
title: "DocumentModelOptions<T>"
description: "Base configuration options for all document AI models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Document.Options`

Base configuration options for all document AI models.

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
| `EffectiveImageSize` | Gets the effective image size with default fallback. |
| `EffectiveInputChannels` | Gets the effective number of input channels with default fallback. |
| `EffectiveLearningRate` | Gets the effective learning rate with default fallback. |
| `EffectiveMaxGradientNorm` | Gets the effective max gradient norm with default fallback. |
| `EffectiveMaxLayoutElements` | Gets the effective max layout elements with default fallback. |
| `EffectiveMaxSequenceLength` | Gets the effective max sequence length with default fallback. |
| `EffectiveNormalizeBoundingBoxes` | Gets the effective bounding box normalization setting with default fallback. |
| `EffectiveNumAttentionHeads` | Gets the effective number of attention heads with default fallback. |
| `EffectiveNumLayers` | Gets the effective number of layers with default fallback. |
| `EffectivePatchSize` | Gets the effective patch size with default fallback. |
| `EffectiveUse2DPositionEmbeddings` | Gets the effective 2D position embeddings setting with default fallback. |
| `EffectiveUseGradientClipping` | Gets the effective gradient clipping setting with default fallback. |
| `EffectiveVocabularySize` | Gets the effective vocabulary size with default fallback. |
| `EffectiveWeightDecay` | Gets the effective weight decay with default fallback. |
| `HiddenDimension` | Gets or sets the hidden dimension size for the model. |
| `ImageSize` | Gets or sets the expected input image size. |
| `InputChannels` | Gets or sets the number of input channels. |
| `LearningRate` | Gets or sets the learning rate for optimization. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for clipping. |
| `MaxLayoutElements` | Gets or sets the maximum number of layout elements to process. |
| `MaxSequenceLength` | Gets or sets the maximum text sequence length. |
| `NormalizeBoundingBoxes` | Gets or sets whether to normalize bounding box coordinates. |
| `NumAttentionHeads` | Gets or sets the number of attention heads for transformer-based models. |
| `NumLayers` | Gets or sets the number of transformer/encoder layers. |
| `PatchSize` | Gets or sets the patch size for vision transformers. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `Use2DPositionEmbeddings` | Gets or sets whether to include 2D position embeddings for layout information. |
| `UseGpu` | Gets or sets whether to use GPU acceleration if available. |
| `UseGradientClipping` | Gets or sets whether to use gradient clipping. |
| `UseMixedPrecision` | Gets or sets whether to use mixed precision (FP16) for faster computation. |
| `VocabularySize` | Gets or sets the vocabulary size for text tokenization. |
| `WeightDecay` | Gets or sets the weight decay for L2 regularization. |

