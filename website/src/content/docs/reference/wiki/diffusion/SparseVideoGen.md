---
title: "SparseVideoGen<T>"
description: "Sparse video generation with selective frame denoising for faster inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Acceleration`

Sparse video generation with selective frame denoising for faster inference.

## For Beginners

Sparse Video Generation achieves ~2x speedup by exploiting spatial-temporal sparsity - skipping computations in regions that change little between frames or across spatial locations.

## How It Works

**References:**

- Paper: "Sparse VideoGen: Accelerating Video Diffusion Transformers with Flexible Sparsity" (2025)

SparseVideoGen accelerates video diffusion inference by identifying and skipping redundant
computations. The key insight is that not all frames require equal denoising effort:

- Keyframes: get full denoising (all transformer blocks)
- Intermediate frames: use sparse computation (skip similar blocks)
- Selection criteria: based on temporal motion magnitude

This achieves 2-4x speedup with minimal visual quality degradation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparseVideoGen(Int32,Int32,Double,SparsityStrategy)` | Initializes a new SparseVideoGen module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KeyframeCount` | Gets the number of keyframes. |
| `KeyframeInterval` | Gets the keyframe interval. |
| `SparsityRatio` | Gets the sparsity ratio (fraction of computation saved). |
| `Strategy` | Gets the sparsity strategy. |
| `TotalFrames` | Gets the total number of frames. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBlocksToSkip(Int32)` | Gets the set of transformer block indices to skip for a non-keyframe. |
| `GetKeyframeIndices` | Gets the indices of all keyframes. |
| `IsKeyframe(Int32)` | Checks if a given frame is a keyframe requiring full computation. |

