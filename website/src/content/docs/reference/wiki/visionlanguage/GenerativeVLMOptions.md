---
title: "GenerativeVLMOptions"
description: "Base configuration options for generative vision-language models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Generative`

Base configuration options for generative vision-language models.

## For Beginners

These options configure the Generative model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GenerativeVLMOptions` | Initializes a new instance with default values. |
| `GenerativeVLMOptions(GenerativeVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ArchitectureType` | Gets or sets the generative architecture type. |
| `DecoderDim` | Gets or sets the text decoder hidden dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ImageMean` | Gets or sets the per-channel image normalization mean. |
| `ImageSize` | Gets or sets the input image size. |
| `ImageStd` | Gets or sets the per-channel image normalization std. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxGenerationLength` | Gets or sets the maximum generation output length in tokens. |
| `MaxSequenceLength` | Gets or sets the maximum input text sequence length. |
| `ModelPath` | Gets or sets the ONNX model path. |
| `NumDecoderLayers` | Gets or sets the number of text decoder layers. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumVisionLayers` | Gets or sets the number of vision encoder layers. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `VisionDim` | Gets or sets the vision encoder feature dimension. |
| `VocabSize` | Gets or sets the vocabulary size for tokenization. |
| `WeightDecay` | Gets or sets the weight decay. |

