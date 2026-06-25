---
title: "EVACLIP<T>"
description: "EVA-CLIP model combining EVA-02 masked image modeling backbone with CLIP contrastive training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

EVA-CLIP model combining EVA-02 masked image modeling backbone with CLIP contrastive training.

## For Beginners

EVA-CLIP improves CLIP by first pre-training the vision encoder
with masked image modeling (learning to reconstruct hidden image patches), then applying
contrastive image-text training. The largest variant has 4.4 billion parameters and achieves
82.0% zero-shot ImageNet accuracy. It also adds rotary position embeddings (RoPE) and SwiGLU
activations for stronger feature learning. Default values follow the original paper
settings.

## How It Works

EVA-CLIP (Sun et al., 2023) uses the EVA-02 ViT backbone pre-trained with masked image modeling (MIM)
for stronger visual representations. The largest variant (ViT-E/14, 4.4B params) achieves 82.0% zero-shot
on ImageNet. EVA-02 adds RoPE positional embeddings and SwiGLU FFN activations.

**References:**

- Paper: "EVA-CLIP: Improved Training Techniques for CLIP at Scale" (Sun et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

