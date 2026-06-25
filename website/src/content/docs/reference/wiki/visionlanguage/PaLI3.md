---
title: "PaLI3<T>"
description: "PaLI-3: efficient PaLI with SigLIP ViT encoder, smaller and better."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

PaLI-3: efficient PaLI with SigLIP ViT encoder, smaller and better.

## For Beginners

PaLI-3 achieves strong vision-language performance with a much
smaller model than PaLI-X by replacing the contrastive ViT encoder with a SigLIP ViT that
uses sigmoid-based contrastive loss instead of softmax. This produces better vision
representations while being significantly more efficient, demonstrating that smarter
pretraining can compensate for smaller model size. Default values follow the original
paper settings.

## How It Works

PaLI-3 (Chen et al., 2023) achieves strong performance with a smaller model by replacing
the contrastive ViT with a SigLIP ViT encoder. The architecture retains the encoder-decoder
design but benefits from the improved vision representations of SigLIP pretraining.

**References:**

- Paper: "PaLI-3 Vision Language Models: Smaller, Faster, Stronger" (Chen et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsurePatchEmbedForParameterVector(Int32)` | Lazily creates _patchEmbed when the incoming parameter vector is longer than the layer-sum, indicating the saved model was trained in vision mode. |
| `GenerateFromImage(Tensor<>,String)` | Generates text using PaLI-3's efficient SigLIP-based architecture. |
| `GetExtraTrainableLayers` | Surfaces _patchEmbed (which lives outside Layers) to the base weight-registry walker so its trainable tensors land in the streaming pool when ConfigureWeightLifetime is called. |

