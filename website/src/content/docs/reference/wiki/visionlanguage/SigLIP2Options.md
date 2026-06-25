---
title: "SigLIP2Options"
description: "Configuration options for SigLIP 2 (Multilingual Vision-Language Encoders with Improved Semantic Understanding)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for SigLIP 2 (Multilingual Vision-Language Encoders with
Improved Semantic Understanding).

## For Beginners

SigLIP 2 is an improved version of SigLIP that learns from three tasks
simultaneously: (1) matching images with text, (2) generating image descriptions, and
(3) predicting hidden parts of images. This multi-task learning produces better representations
and supports many languages.

## How It Works

SigLIP 2 (Tschannen et al., 2025) improves upon SigLIP by combining multiple training
objectives: sigmoid contrastive loss, autoregressive captioning loss, and self-supervised
masked image modeling. This multi-objective approach produces encoders with better semantic
understanding while maintaining efficient scaling. The model also supports 32+ languages
through a multilingual text encoder.

**Key Differences from SigLIP:**

- **Multi-objective training**: Contrastive + captioning + self-supervised losses
- **Captioning decoder**: Lightweight autoregressive decoder for generating image descriptions
- **Masked image modeling**: Self-supervised patch prediction for better spatial understanding
- **Multilingual**: Extended vocabulary with 32+ language support via mPaLM tokenizer
- **Online data curation**: Dynamic mixing of data sources during training

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SigLIP2Options` | Initializes default SigLIP 2 options. |
| `SigLIP2Options(SigLIP2Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CaptioningDecoderDim` | Gets or sets the hidden dimension for the captioning decoder. |
| `CaptioningLossWeight` | Gets or sets the weight for the captioning loss in multi-objective training. |
| `IncludeCaptioningDecoder` | Gets or sets whether to include the captioning decoder in inference mode. |
| `LossType` | Gets or sets the contrastive loss type (default: Sigmoid for SigLIP family). |
| `MaxCaptionLength` | Gets or sets the maximum caption length for the captioning decoder. |
| `MimDecoderDim` | Gets or sets the MIM decoder dimension for predicting masked patch features. |
| `MimMaskRatio` | Gets or sets the mask ratio for masked image modeling (fraction of patches masked). |
| `Multilingual` | Gets or sets whether multilingual text encoding is enabled. |
| `NumCaptioningDecoderHeads` | Gets or sets the number of attention heads in the captioning decoder. |
| `NumCaptioningDecoderLayers` | Gets or sets the number of captioning decoder layers. |
| `NumMimDecoderLayers` | Gets or sets the number of MIM decoder layers. |
| `SelfSupervisedLossWeight` | Gets or sets the weight for the self-supervised masked image modeling loss. |
| `SigmoidBias` | Gets or sets the bias term for sigmoid contrastive loss. |

