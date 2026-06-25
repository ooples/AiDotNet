---
title: "MetaDMOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-DM: Applications of Diffusion Models on Few-Shot Learning (Hu et al., ICIP 2024)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-DM: Applications of Diffusion Models on Few-Shot Learning
(Hu et al., ICIP 2024).

## How It Works

Meta-DM uses a DDPM-style diffusion model as a data augmentation module for few-shot learning.
It generates synthetic support samples conditioned on the existing few-shot support set, then
trains on the enriched dataset. This is a modular augmentation strategy composable with any
gradient-based meta-learning algorithm.

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Ending noise schedule parameter. |
| `BetaStart` | Starting noise schedule parameter. |
| `DenoisingSteps` | Number of denoising steps for generation (≤ DiffusionTimesteps). |
| `DiffusionTimesteps` | Number of diffusion timesteps for generation. |
| `MatchingWeight` | Weight of the distribution matching loss relative to the task loss. |
| `PrototypeDim` | Dimensionality of the prototype embeddings for distribution matching. |
| `SyntheticSamplesPerClass` | Number of synthetic samples to generate per class for augmentation. |

