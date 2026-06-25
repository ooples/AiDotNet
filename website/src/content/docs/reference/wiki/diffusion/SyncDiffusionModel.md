---
title: "SyncDiffusionModel<T>"
description: "SyncDiffusion model for coherent panorama generation with synchronized denoising."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Panorama`

SyncDiffusion model for coherent panorama generation with synchronized denoising.

## For Beginners

SyncDiffusion creates panoramas that look more consistent
than MultiDiffusion. While MultiDiffusion just averages overlapping patches,
SyncDiffusion adds an extra step to make sure all patches agree on the overall
style and color, preventing abrupt style changes across the panorama.

## How It Works

SyncDiffusion improves on MultiDiffusion by adding a gradient descent synchronization
step that ensures global style consistency across patches. After each denoising step,
patches are updated to minimize perceptual difference with a shared anchor view.

Reference: Lee et al., "SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions", NeurIPS 2023

