---
title: "MMDiTXNoisePredictor<T>"
description: "Extended Multimodal Diffusion Transformer (MMDiT-X) noise predictor for the Stable Diffusion 3.5 architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Extended Multimodal Diffusion Transformer (MMDiT-X) noise predictor for the
Stable Diffusion 3.5 architecture.

## For Beginners

This is the transformer behind Stable Diffusion 3.5.
Image patches and text tokens look at each other in one shared attention step,
and the diffusion timestep steers every layer. Pick a size with the variant.

## How It Works

MMDiT-X is the SD3.5 evolution of MMDiT (Esser et al. 2024): dual image/text
streams with joint (concatenated) self-attention, separate per-stream MLPs,
adaptive-LayerNorm timestep modulation, and QK-normalization. It is realized on
the faithful `MMDiTNoisePredictor` dual-stream backbone at the
SD3.5 scale selected by `MMDiTXVariant` (Medium: hidden 2048 / 24
blocks / 16 heads; Large & LargeTurbo: hidden 2560 / 38 blocks / 20 heads).
This replaces the previous Dense-only placeholder that had no attention, no
timestep conditioning, and ignored the text context.

Reference: Esser et al., "Scaling Rectified Flow Transformers for
High-Resolution Image Synthesis", ICML 2024 (arXiv:2403.03206).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MMDiTXNoisePredictor(MMDiTXVariant,Int32,Int32,Int32,Nullable<Int32>,Int32,Int32,Int32)` | Initializes a new MMDiT-X (SD3.5) predictor on the faithful MMDiT dual-stream backbone at the scale selected by `variant`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |

