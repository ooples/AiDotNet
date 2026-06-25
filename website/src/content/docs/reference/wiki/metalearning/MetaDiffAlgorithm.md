---
title: "MetaDiffAlgorithm<T, TInput, TOutput>"
description: "Implementation of MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning (Zhang et al., AAAI 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning
(Zhang et al., AAAI 2024).

## How It Works

MetaDiff replaces the gradient-based inner loop of MAML with a learned diffusion-based
denoising process. Starting from Gaussian noise, a task-conditional denoising network
(TCUNet) iteratively produces task-specific weight parameters. The key insight is that
gradient descent (random init → optimal weights) is analogous to the diffusion reverse
process (noise → clean signal).

**Algorithm:**

**Key advantage:** GPU memory is constant regardless of adaptation steps
(unlike MAML which scales linearly). The denoising process can be run for as many
steps as desired without backpropagating through the full chain.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeTargetDelta(Vector<>,IMetaLearningTask<,,>)` | Computes target parameter delta by gradient adaptation on support set (compressed). |
| `ComputeTaskCondition()` | Computes task conditioning vector from support features. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `PredictNoise(Double[],Vector<>,Int32)` | Denoiser MLP: predicts noise given noised weights, task condition, and timestep. |
| `RunDenoisingInference(Vector<>,Int32)` | Runs the reverse diffusion process (denoising) to generate task-specific weight deltas. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alphas` | α_t = 1 - β_t. |
| `_alphasCumprod` | ᾱ_t = cumulative product of α. |
| `_betas` | Noise schedule: β_t values. |
| `_denoiserParams` | Denoiser network parameters (task-conditional noise predictor). |
| `_taskEncoderParams` | Task encoder parameters: maps support features to conditioning vector. |

