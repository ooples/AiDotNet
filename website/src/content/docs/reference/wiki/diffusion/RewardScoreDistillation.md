---
title: "RewardScoreDistillation<T>"
description: "Reward-weighted Score Distillation Sampling (RewardSDS) for preference-aligned 3D generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Reward-weighted Score Distillation Sampling (RewardSDS) for preference-aligned 3D generation.

## For Beginners

Standard SDS creates realistic-looking 3D objects, but they may not
match what people actually want aesthetically. RewardSDS adds a "quality judge" (reward model)
that steers the generation toward more visually appealing results. It's like having both a
realism expert and an art critic guiding the 3D artist simultaneously.

## How It Works

RewardSDS augments standard SDS gradients with reward model feedback. Instead of using
only the diffusion model's score, it weights the gradient by a reward signal that measures
alignment with human preferences (aesthetics, text-image correspondence, etc.). This
produces 3D objects that not only look realistic but also match desired aesthetic qualities.

Reference: Adapted from DreamReward (Ye et al., 2024) combining human preference alignment
with score distillation for text-to-3D generation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RewardScoreDistillation(IDiffusionModel<>,Double,Double)` | Initializes a new RewardSDS instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets the guidance scale for score distillation. |
| `RewardWeight` | Gets the reward signal weight. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>,Vector<>,Double)` | Computes the reward-weighted SDS gradient. |

