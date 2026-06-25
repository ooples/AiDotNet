---
title: "DiffusionDPO<T>"
description: "Direct Preference Optimization (DPO) adapted for diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Alignment`

Direct Preference Optimization (DPO) adapted for diffusion models.

## For Beginners

DPO teaches the model by showing it pairs of images: one that
humans preferred and one they didn't. Instead of training a separate "judge" model first
(like RLHF does), DPO directly adjusts the diffusion model to produce more of what
humans like. It's simpler and often more stable than RLHF for alignment.

## How It Works

Diffusion-DPO adapts the DPO framework from language models to diffusion models.
Given pairs of preferred and dispreferred images, it directly optimizes the diffusion
model's policy to prefer generating the preferred outputs without needing a separate
reward model.

Reference: Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionDPO(IDiffusionModel<>,IDiffusionModel<>,Double,Double)` | Initializes a new Diffusion-DPO trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta` | Gets the beta temperature parameter. |
| `LabelSmoothing` | Gets the label smoothing factor. |
| `Model` | Gets the model being aligned. |
| `ReferenceModel` | Gets the frozen reference model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDPOLoss(,,,)` | Computes the DPO loss given preferred and dispreferred noise predictions. |
| `ComputeImplicitReward(,)` | Computes the implicit reward for a sample. |

