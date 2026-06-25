---
title: "AsymmDiTPredictor<T>"
description: "Asymmetric Diffusion Transformer (AsymmDiT) noise predictor for video generation (Genmo Mochi 1 architecture)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Asymmetric Diffusion Transformer (AsymmDiT) noise predictor for video
generation (Genmo Mochi 1 architecture).

## How It Works

AsymmDiT (Mochi 1) is an MMDiT-family transformer: it jointly attends to text
and visual tokens with multi-modal self-attention and learns separate per-stream
MLPs, with adaptive-LayerNorm timestep modulation — exactly the MMDiT block.
It is therefore realized on the faithful `MMDiTNoisePredictor`
dual-stream backbone (joint concatenated attention + per-stream MLPs + adaLN),
replacing the previous Dense-only placeholder that had no attention or timestep
conditioning.

**Known deviation from Mochi:** Mochi's defining feature is the
*asymmetry* — the visual stream carries ~4× the parameters of the text
stream (a wider visual hidden dim) via non-square QKV/output projections. The
MMDiT backbone here is *symmetric* (both streams share the hidden width),
so this is a faithful realization of Mochi's joint dual-stream MMDiT block but
not yet its stream-width asymmetry. Full asymmetry is tracked as a backbone
enhancement (non-square per-stream projections) rather than reverting to the
non-faithful Dense stub.

Reference: Genmo, "Mochi 1: A New SOTA in Open-Source Video Generation", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AsymmDiTPredictor(Int32,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new AsymmDiT (Mochi) predictor on the faithful MMDiT dual-stream backbone. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |

