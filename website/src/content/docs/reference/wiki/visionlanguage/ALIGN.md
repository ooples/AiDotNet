---
title: "ALIGN<T>"
description: "ALIGN (A Large-scale ImaGe and Noisy-text embedding) model for zero-shot classification and cross-modal retrieval using EfficientNet as the vision encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

ALIGN (A Large-scale ImaGe and Noisy-text embedding) model for zero-shot classification
and cross-modal retrieval using EfficientNet as the vision encoder.

## For Beginners

ALIGN is similar to CLIP but uses a different type of image model
(EfficientNet, which is a convolutional neural network) instead of a Vision Transformer.
It was trained on a very large but noisy dataset, proving that more data beats cleaner data.

## How It Works

ALIGN (Jia et al., ICML 2021) demonstrates that simple dual-encoder contrastive learning
achieves strong image-text alignment when trained at massive scale. Unlike CLIP which uses
a Vision Transformer, ALIGN uses an EfficientNet-B7 as its vision encoder, showing that
the contrastive learning recipe is architecture-agnostic.

**Key Innovation:** ALIGN trains on 1.8 billion noisy alt-text image-text pairs without
expensive filtering, showing that scale compensates for data noise. The EfficientNet backbone
provides a CNN-based alternative to ViT for vision encoding.

**References:**

- Paper: "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision" (Jia et al., ICML 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ALIGN(NeuralNetworkArchitecture<>,ALIGNOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an ALIGN model in native training mode. |
| `ALIGN(NeuralNetworkArchitecture<>,String,ALIGNOptions)` | Creates an ALIGN model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |
| `SyncImageSizeWithArchitecture` | Aligns `_options.ImageSize` with `Architecture.InputHeight` when the architecture declares an explicit square spatial extent. |

