---
title: "EMMDiTPredictor<T>"
description: "E-MMDiT (Efficient Multimodal Diffusion Transformer) noise predictor — a compact configuration of the MMDiT architecture (Stable Diffusion 3, Esser et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

E-MMDiT (Efficient Multimodal Diffusion Transformer) noise predictor — a
compact configuration of the MMDiT architecture (Stable Diffusion 3,
Esser et al. 2024) for parameter-efficient inference.

## For Beginners

This is a "small but real" version of the Stable
Diffusion 3 transformer. It does exactly what the big one does — image and
text tokens attend to each other jointly, and the timestep controls every
layer — just with fewer/narrower layers so it runs lighter.

## How It Works

E-MMDiT keeps the full MMDiT block — dual image/text streams with joint
(concatenated) self-attention, separate per-stream MLPs, adaptive-LayerNorm
modulation from the timestep + pooled-text conditioning, and QK-normalization
— but at a smaller width/depth (hidden 1024, 12 joint blocks, 16 heads) than
the full SD3 MMDiT. It is therefore realized as a configured
`MMDiTNoisePredictor`: the architecture is identical to MMDiT,
only the scale differs. (The earlier Dense-only placeholder had no attention
or timestep conditioning and was not the MMDiT architecture it claimed.)

Reference: Esser et al., "Scaling Rectified Flow Transformers for
High-Resolution Image Synthesis", ICML 2024 (arXiv:2403.03206).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EMMDiTPredictor(Int32,Int32,Nullable<Int32>)` | Initializes a new E-MMDiT predictor on the faithful MMDiT backbone at the compact (hidden 1024, 12 joint blocks, 16 heads) configuration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |

