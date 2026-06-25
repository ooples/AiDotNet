---
title: "FoundationalVLMOptions"
description: "Base configuration options for foundational vision-language fusion models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Base configuration options for foundational vision-language fusion models.

## For Beginners

These options configure the Foundational model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FoundationalVLMOptions` | Initializes a new instance with default values. |
| `FoundationalVLMOptions(FoundationalVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FusionDim` | Gets or sets the fusion/hidden dimension. |
| `FusionType` | Gets or sets the fusion type. |
| `ImageMean` | Gets or sets the per-channel image normalization mean. |
| `ImageSize` | Gets or sets the input image size. |
| `ImageStd` | Gets or sets the per-channel image normalization std. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxSequenceLength` | Gets or sets the maximum text sequence length. |
| `ModelPath` | Gets or sets the ONNX model path. |
| `NumFusionLayers` | Gets or sets the number of cross-modal/fusion layers. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumTextLayers` | Gets or sets the number of text encoder layers. |
| `NumVisionLayers` | Gets or sets the number of vision encoder layers. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `TextDim` | Gets or sets the text embedding dimension. |
| `VisionDim` | Gets or sets the vision feature dimension. |
| `VisualFeatureType` | Gets or sets the visual feature type. |
| `VocabSize` | Gets or sets the vocabulary size. |
| `WeightDecay` | Gets or sets the weight decay. |

