---
title: "TTVSR<T>"
description: "TTVSR: learning trajectory-aware transformer for long-range video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

TTVSR: learning trajectory-aware transformer for long-range video super-resolution.

## For Beginners

Imagine following a ball as it moves across frames. Instead of
looking at the same fixed spot in every frame (which would miss the ball after it
moves), TTVSR tracks where objects actually go and gathers information along their
travel path. This is much more effective because real video has complex motion, and
the best information for upscaling a pixel often comes from a completely different
position in neighboring frames.

**Usage:**

## How It Works

TTVSR (Liu et al., ECCV 2022) learns trajectory-aware features for temporal modeling:

- Trajectory-aware attention: instead of attending to fixed spatial locations across

frames, attention follows estimated motion trajectories so features are gathered along
the actual path each visual element traveled

- Cross-scale feature tokenization: visual tokens are extracted at multiple spatial

scales and fused, capturing both fine textures and coarse structures simultaneously

- Location map: a learned spatial routing map that helps the transformer locate the

correct trajectory positions across the full video sequence

- Long-range modeling: trajectories span the full video, not just adjacent frames,

enabling information flow from temporally distant frames along motion paths

**Reference:** "TTVSR: Learning Trajectory-Aware Transformer for Video
Super-Resolution" (Liu et al., ECCV 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TTVSR(NeuralNetworkArchitecture<>,String,TTVSROptions)` | Creates a TTVSR model in ONNX inference mode. |
| `TTVSR(NeuralNetworkArchitecture<>,TTVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a TTVSR model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

