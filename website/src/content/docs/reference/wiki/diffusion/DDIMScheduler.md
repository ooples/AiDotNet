---
title: "DDIMScheduler<T>"
description: "DDIM (Denoising Diffusion Implicit Models) scheduler implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

DDIM (Denoising Diffusion Implicit Models) scheduler implementation.

## For Beginners

DDIM is like a shortcut for removing noise from images.

Imagine you have a very blurry photo and need to make it clear:

- DDPM (original method): Take 1000 tiny steps to slowly reveal the image
- DDIM (this method): Take 50 larger steps to reveal the image faster

The magic is the "eta" parameter:

- eta=0: Deterministic - same input always produces same output (faster, consistent)
- eta=1: Stochastic - adds randomness like DDPM (slower, more variety)
- eta between 0-1: Mix of both behaviors

Key advantages of DDIM:

- Much faster generation (10-50x fewer steps needed)
- Deterministic option allows reproducible results
- Can interpolate smoothly between images (useful for animations)

## How It Works

DDIM is a faster variant of DDPM that can achieve similar quality with far fewer
denoising steps. While DDPM requires many steps (often 1000), DDIM can achieve
similar quality with 50 or fewer steps by using a different mathematical formulation.

**Reference:** "Denoising Diffusion Implicit Models" by Song et al., 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDIMScheduler(SchedulerConfig<>)` | Initializes a new instance of the DDIM scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one DDIM denoising step. |

