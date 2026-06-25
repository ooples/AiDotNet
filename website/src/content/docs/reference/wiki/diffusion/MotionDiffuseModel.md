---
title: "MotionDiffuseModel<T>"
description: "MotionDiffuse model for fine-grained text-driven motion generation with body part control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MotionGeneration`

MotionDiffuse model for fine-grained text-driven motion generation with body part control.

## For Beginners

MotionDiffuse gives you more control over body animation than
MDM. You can describe what each body part should do separately — like "wave the right
hand while walking forward" — and each part follows its own instruction precisely.

## How It Works

MotionDiffuse provides fine-grained control over body part motions through separate
text conditioning for different body segments (arms, legs, torso). It uses a
part-aware cross-attention mechanism to bind text descriptions to specific body parts.

Reference: Zhang et al., "MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model", 2024

## Fields

| Field | Summary |
|:-----|:--------|
| `MOTION_FEATURE_DIM` | Number of motion feature dimensions per frame (263 = 3 root velocity + 6*N joint rotations + ...). |

