---
title: "D3PO<T>"
description: "Direct Preference for Denoising Diffusion Policy Optimization (D3PO)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Alignment`

Direct Preference for Denoising Diffusion Policy Optimization (D3PO).

## For Beginners

Regular DPO teaches a model by comparing two finished images. D3PO
goes deeper — it compares images at every step of the generation process. This is like
having a coach who gives feedback on every brush stroke rather than just rating the final
painting, leading to more precise control over image quality.

## How It Works

D3PO extends DPO to operate directly on the denoising process of diffusion models.
Rather than comparing final generated images, it applies preference optimization at each
denoising step, treating the denoising trajectory as a Markov Decision Process. This
enables finer-grained alignment by optimizing the generation policy at the step level.

Reference: Yang et al., "Using Human Feedback to Fine-tune Diffusion Models without Any
Reward Model", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `D3PO(IDiffusionModel<>,IDiffusionModel<>,Double,Double)` | Initializes a new D3PO trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta` | Gets the temperature parameter beta. |
| `StepWeight` | Gets the per-step weighting factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeStepLoss(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes the D3PO per-step loss at a single denoising timestep. |

