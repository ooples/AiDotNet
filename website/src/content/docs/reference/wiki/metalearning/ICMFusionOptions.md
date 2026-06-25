---
title: "ICMFusionOptions<T, TInput, TOutput>"
description: "Configuration options for ICM-Fusion (In-Context Meta-Optimized LoRA Fusion, 2025)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for ICM-Fusion (In-Context Meta-Optimized LoRA Fusion, 2025).

## How It Works

ICM-Fusion fuses multiple task-specific parameter deltas by encoding them into a latent
space via a Fusion-VAE, then reconstructing the fused adapter. The VAE is meta-learned
so that task vector arithmetic in latent space resolves inter-weight conflicts.

## Properties

| Property | Summary |
|:-----|:--------|
| `FusionDecay` | Exponential decay for older fusion components. |
| `KLWeight` | KL divergence weight in the VAE loss: L = L_recon + KLWeight * L_KL. |
| `LatentDim` | Dimensionality of the VAE latent space. |
| `NumFusionComponents` | Number of task-specific components to maintain for fusion. |

