---
title: "MetaDDPMAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models.

## How It Works

Meta-DDPM meta-learns a denoising diffusion model that generates task-specific model
weights conditioned on support set embeddings. It uses the full DDPM framework with
a linear noise schedule and EMA parameter averaging for stable generation.

**Algorithm:**

**Key difference from MetaDiff:** Meta-DDPM uses the standard DDPM sampling
algorithm with full variance (σ_t = √β_t), while MetaDiff uses the deterministic
DDIM-style denoising. Meta-DDPM also includes EMA for denoiser parameter stability.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |
| `SampleDDPM(Vector<>,Int32)` | Full DDPM sampling (reverse process) with standard variance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_denoiserEma` | EMA copy of denoiser parameters for stable generation. |
| `_denoiserParams` | Denoiser (noise predictor) parameters. |
| `_taskEncoderParams` | Task encoder parameters. |

