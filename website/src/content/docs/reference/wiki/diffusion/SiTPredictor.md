---
title: "SiTPredictor<T>"
description: "Scalable Interpolant Transformer (SiT) noise predictor for flow-based diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Scalable Interpolant Transformer (SiT) noise predictor for flow-based diffusion.

## For Beginners

SiT is "DiT with a more flexible noise process." The neural
network that predicts the noise is exactly the DiT transformer; what changes is
the math used to add/remove noise during training and sampling, which the
scheduler handles. Reusing the DiT backbone here means SiT gets real
self-attention, real timestep conditioning (AdaLN), and the MLP expansion of a
proper transformer — not a placeholder.

## How It Works

SiT (Ma et al., ECCV 2024) deliberately reuses the **DiT backbone unchanged**
— the paper's §3 states it adopts the DiT architecture and isolates its
contribution to the *training/sampling* side (a learnable interpolant
between data and noise that unifies score-based diffusion and flow matching).
The network itself is the identical patchify → adaptive-LayerNorm
self-attention transformer stack → final AdaLN layer. We therefore realize SiT
faithfully as a configured `DiTNoisePredictor`: the interpolant
difference lives in the diffusion model's scheduler (flow-matching /
stochastic-interpolant), not in this noise-prediction network.

Reference: Ma et al., "SiT: Exploring Flow and Diffusion-based Generative Models
with Scalable Interpolant Transformers", ECCV 2024 (arXiv:2401.08740).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SiTPredictor(Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new SiT predictor on the faithful DiT backbone. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |

