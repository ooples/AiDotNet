---
title: "DDPMScheduler<T>"
description: "DDPM (Denoising Diffusion Probabilistic Models) scheduler implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

DDPM (Denoising Diffusion Probabilistic Models) scheduler implementation.

## For Beginners

DDPM is the foundational method for diffusion image generation.

Think of it like restoring a photograph that has been progressively damaged:

- Training: Learn how each level of damage looks
- Generation: Start with pure static and remove damage one tiny step at a time

Key characteristics:

- Stochastic: Each step adds a small amount of random noise
- Many steps needed: Typically 1000 steps for good quality
- Well-studied: Strong theoretical guarantees on output quality
- Variance can be learned or fixed

The step formula:
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_cumprod_t)) * eps) + sigma_t * z

Where z is random noise and sigma_t controls the stochasticity.

## How It Works

DDPM is the original diffusion model scheduler that uses a Markov chain of Gaussian
transitions to gradually denoise samples. It requires many steps (typically 1000) but
produces high-quality results with well-understood theoretical properties.

**Reference:** Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDPMScheduler(SchedulerConfig<>)` | Initializes a new instance of the DDPM scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one DDPM denoising step. |

