---
title: "DiffusionRLHF<T>"
description: "Reinforcement Learning from Human Feedback (RLHF) adapted for diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Alignment`

Reinforcement Learning from Human Feedback (RLHF) adapted for diffusion models.

## For Beginners

RLHF is a two-step process: first, train a "reward model" that
learns what humans prefer (like a judge). Then use that judge to give feedback to the
diffusion model during training. The diffusion model learns to generate images the
judge would rate highly. KL regularization prevents the model from "cheating" by
exploiting the judge's blind spots.

## How It Works

Diffusion-RLHF uses a trained reward model to provide feedback signals that guide
the diffusion model toward generating outputs aligned with human preferences. The
reward model is trained on human preference data, then used to fine-tune the diffusion
model via policy gradient methods with KL regularization against a reference model.

Reference: Black et al., "Training Diffusion Models with Reinforcement Learning", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionRLHF(IDiffusionModel<>,IDiffusionModel<>,Double,Double,Double)` | Initializes a new Diffusion-RLHF trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClipRange` | Gets the PPO clipping range. |
| `KLWeight` | Gets the KL divergence weight. |
| `Model` | Gets the model being aligned. |
| `ReferenceModel` | Gets the frozen reference model. |
| `RewardScale` | Gets the reward scaling factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePPOLoss(,)` | Computes the PPO-clipped policy gradient loss for diffusion denoising steps. |
| `ComputeRLHFObjective(,,)` | Computes the RLHF objective: reward minus KL penalty. |

