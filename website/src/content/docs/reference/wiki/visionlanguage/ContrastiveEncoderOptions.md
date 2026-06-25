---
title: "ContrastiveEncoderOptions"
description: "Base configuration options for contrastive vision-language encoders (CLIP-family models)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Base configuration options for contrastive vision-language encoders (CLIP-family models).

## For Beginners

These settings control how the model processes images and text.
The most important ones are:

- **ImageSize**: The size images are resized to before processing (e.g., 224x224 pixels)
- **EmbeddingDim**: How many numbers represent each image or text (bigger = more detail)
- **Temperature**: How "sharp" the similarity comparisons are (lower = more decisive)

## How It Works

Contrastive encoders learn a shared embedding space for images and text via contrastive learning.
They share common hyperparameters: image/text embedding dimensions, projection dimension,
temperature, and image preprocessing settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContrastiveEncoderOptions` | Initializes a new instance with default values. |
| `ContrastiveEncoderOptions(ContrastiveEncoderOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ImageEncoderModelPath` | Gets or sets the path to the image encoder ONNX model. |
| `ImageMean` | Gets or sets the per-channel mean for image normalization. |
| `ImageSize` | Gets or sets the input image size (height = width). |
| `ImageStd` | Gets or sets the per-channel standard deviation for image normalization. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `MaxSequenceLength` | Gets or sets the maximum text token sequence length. |
| `NumTextHeads` | Gets or sets the number of text attention heads. |
| `NumTextLayers` | Gets or sets the number of text transformer layers. |
| `NumVisionHeads` | Gets or sets the number of vision attention heads. |
| `NumVisionLayers` | Gets or sets the number of vision transformer layers. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PatchSize` | Gets or sets the ViT patch size in pixels. |
| `ProjectionDim` | Gets or sets the shared projection dimension for the joint embedding space. |
| `Temperature` | Gets or sets the temperature parameter for contrastive loss. |
| `TextEmbeddingDim` | Gets or sets the text encoder embedding dimension. |
| `TextEncoderModelPath` | Gets or sets the path to the text encoder ONNX model. |
| `TextEncoderVariant` | Gets or sets the text encoder variant. |
| `VisionEmbeddingDim` | Gets or sets the vision encoder embedding dimension. |
| `VisionEncoderVariant` | Gets or sets the vision encoder variant. |
| `VisionFfnMultiplier` | Gets or sets the feed-forward hidden dimension multiplier for vision encoder. |
| `VocabSize` | Gets or sets the vocabulary size. |
| `WarmUpSteps` | Gets or sets the warm-up steps for the learning rate scheduler. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

