---
title: "BMAMLOptions<T, TInput, TOutput>"
description: "Configuration options for BMAML: Bayesian Model-Agnostic Meta-Learning (Yoon et al., NeurIPS 2018)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for BMAML: Bayesian Model-Agnostic Meta-Learning
(Yoon et al., NeurIPS 2018).

## How It Works

BMAML uses Stein Variational Gradient Descent (SVGD) to maintain a set of
particles (parameter vectors) that approximate the posterior over task-specific
parameters. Instead of a single point estimate (as in MAML), BMAML produces
a particle ensemble for uncertainty-aware predictions.

**Key equations:**

## Properties

| Property | Summary |
|:-----|:--------|
| `KernelBandwidth` | RBF kernel bandwidth. |
| `NumParticles` | Number of particles (ensemble members) for SVGD posterior approximation. |
| `ParticleInitScale` | Scale of Gaussian noise for initial particle perturbation from θ_0. |
| `SVGDRepulsiveWeight` | Weight for the SVGD repulsive (entropy) term relative to the attractive (likelihood) term. |

