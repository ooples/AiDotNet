---
title: "FluxDoubleStreamPredictor<T>"
description: "FLUX.1 double-stream transformer noise predictor (Black Forest Labs)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

FLUX.1 double-stream transformer noise predictor (Black Forest Labs).

## For Beginners

FLUX first lets image and text tokens interact in two
separate "lanes" that still attend to each other (double-stream), then merges
them into one lane for deeper fusion (single-stream). Every layer is steered by
the diffusion timestep. This is the real architecture, not a stand-in.

## How It Works

FLUX.1 is a 12B rectified-flow transformer with a hybrid MMDiT design:
**19 double-stream blocks** (separate image/text streams with joint
concatenated self-attention and per-stream MLPs, à la SD3 MMDiT) followed by
**38 single-stream blocks** (text and image concatenated into one sequence
through a unified path), at hidden size 3072 with 24 attention heads and
adaptive-LayerNorm timestep modulation throughout. It is realized on the
faithful `MMDiTNoisePredictor` backbone, which natively supports
the joint-block + single-block split (numJointLayers + numSingleLayers). This
replaces the previous Dense-only placeholder that had no attention, no
timestep conditioning, and no dual stream.

Reference: Black Forest Labs, "FLUX.1", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FluxDoubleStreamPredictor(FluxPredictorVariant,Int32,Int32,Nullable<Int32>)` | Initializes a new FLUX.1 predictor on the faithful MMDiT dual-stream backbone (19 joint + 38 single blocks, hidden 3072, 24 heads). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |

