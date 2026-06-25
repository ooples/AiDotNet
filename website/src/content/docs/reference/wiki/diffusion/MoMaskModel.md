---
title: "MoMaskModel<T>"
description: "MoMask model for masked generative modeling of 3D human motion sequences."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MotionGeneration`

MoMask model for masked generative modeling of 3D human motion sequences.

## For Beginners

MoMask generates human motion faster than diffusion-based methods
by converting motion into tokens (like words in a sentence) and predicting masked tokens
in parallel. This is similar to how BERT fills in missing words, but for body movement.

## How It Works

MoMask generates human motion using masked token prediction in a discrete motion
token space. It first quantizes motion into tokens via RVQ (residual vector quantization),
then uses masked prediction for fast parallel generation.

Reference: Guo et al., "MoMask: Generative Masked Modeling of 3D Human Motions", CVPR 2024

## Fields

| Field | Summary |
|:-----|:--------|
| `MOTION_FEATURE_DIM` | Number of motion feature dimensions per frame (263 = 3 root velocity + 6*N joint rotations + ...). |

