---
title: "UViTNoisePredictor<T>"
description: "U-shaped Vision Transformer (U-ViT) noise predictor for diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

U-shaped Vision Transformer (U-ViT) noise predictor for diffusion models.

## For Beginners

U-ViT is like a transformer with U-Net-style shortcuts:

U-Net: Uses conv layers with skip connections
DiT: Uses transformer layers without skip connections
U-ViT: Uses transformer layers WITH skip connections (best of both)

Architecture:

1. Patchify: Split image into patches
2. Encoder transformer blocks (L/2 blocks)
3. Middle transformer block
4. Decoder transformer blocks (L/2 blocks) with skip connections from encoder
5. Unpatchify: Reconstruct output

Used in: UniDiffuser (multi-modal generation).

## How It Works

U-ViT combines the best of U-Net and Vision Transformer architectures.
It applies a transformer to image patches but adds long skip connections
between encoder and decoder blocks, similar to U-Net.

Reference: Bao et al., "All are Worth Words: A ViT Backbone for Diffusion Models", CVPR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UViTNoisePredictor(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the U-ViT noise predictor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` |  |
| `ContextDimension` |  |
| `HiddenSize` | Gets the hidden dimension of the transformer. |
| `InputChannels` |  |
| `OutputChannels` |  |
| `ParameterCount` |  |
| `PatchSize` | Gets the patch size used for tokenizing spatial features. |
| `SupportsCFG` |  |
| `SupportsCrossAttention` |  |
| `TimeEmbeddingDim` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetParameterChunks` |  |
| `GetParameters` |  |
| `PredictNoise(Tensor<>,Int32,Tensor<>)` |  |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` |  |
| `SetParameterChunks(IEnumerable<Tensor<>>)` |  |
| `SetParameters(Vector<>)` |  |
| `UViTLayerSequence` | The full non-null layer list in canonical serialization order (patch/time embeds, encoder blocks, middle block, each decoder block followed by its skip projection, final norm, output projection). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_maxPatches` | Maximum number of patches (computed from latent size and patch size). |

