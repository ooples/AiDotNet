---
title: "EVACLIPOptions"
description: "Configuration options for the EVA-CLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the EVA-CLIP model.

## For Beginners

EVA-CLIP is like CLIP but with a better-trained image encoder. The EVA
vision model first learns to understand images on its own (by filling in masked patches), and
then learns to connect images with text. This two-step process produces stronger results.

## How It Works

EVA-CLIP (Sun et al., 2023) combines the EVA-02 Vision Transformer backbone with CLIP-style
contrastive learning. EVA-02 uses masked image modeling (MIM) pre-training to produce a
stronger vision encoder, which is then used for contrastive image-text alignment. The largest
EVA-CLIP variant (ViT-E/14, 4.4B params) achieves 82.0% zero-shot on ImageNet.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EVACLIPOptions` | Initializes default EVA-CLIP options. |
| `EVACLIPOptions(EVACLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LossType` | Gets or sets the contrastive loss type. |
| `MIMPretraining` | Gets or sets the MIM pre-training method used for the vision encoder. |
| `UseEVA02` | Gets or sets whether to use EVA-02 (improved) backbone instead of EVA-01. |
| `UseRoPE` | Gets or sets whether to use rotary position embeddings (RoPE) in the vision encoder. |
| `UseSwiGLU` | Gets or sets whether to use SwiGLU activation in the FFN layers. |

