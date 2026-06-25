---
title: "LCMScheduler<T>"
description: "LCM (Latent Consistency Model) scheduler for ultra-fast diffusion sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

LCM (Latent Consistency Model) scheduler for ultra-fast diffusion sampling.

## For Beginners

LCM is the fastest way to generate images with diffusion models.

The key insight:

- Normal diffusion: Needs 20-50 steps, each requiring a full model evaluation
- LCM: Needs only 2-8 steps by training the model to "skip ahead"

How it achieves this:

1. A teacher model (e.g., Stable Diffusion) is trained normally
2. The LCM student learns to predict what the teacher would produce after many steps
3. At inference, the student can jump directly to near-final results

Key characteristics:

- Ultra-fast: 2-8 steps for good quality (vs 20-50 for normal methods)
- Compatible: Can be applied to existing Stable Diffusion models via LoRA
- Quality: Slight trade-off vs full-step methods, but excellent for interactive use
- Real-time: Enables near-real-time image generation

Common configurations:

- 4 steps with guidance 1.0: Fast, good quality
- 8 steps with guidance 1.5: Higher quality, still very fast

## How It Works

The LCM scheduler implements the sampling procedure for Latent Consistency Models,
which can generate high-quality images in just 1-8 steps. It uses a consistency
distillation approach where the model learns to directly predict the final output.

**Reference:** Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LCMScheduler(SchedulerConfig<>,Int32)` | Initializes a new instance of the LCM scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindTimestepIndex(Int32)` | Finds the index of a timestep in the current schedule. |
| `GetState` |  |
| `SetTimesteps(Int32)` | Sets up the inference timesteps for LCM's skipping schedule. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one LCM denoising step. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_originalInferenceSteps` | The number of original inference steps that each LCM step "skips" through. |

