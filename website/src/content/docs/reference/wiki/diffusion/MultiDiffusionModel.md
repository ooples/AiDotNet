---
title: "MultiDiffusionModel<T>"
description: "MultiDiffusion model for generating seamless panoramic and ultra-wide images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Panorama`

MultiDiffusion model for generating seamless panoramic and ultra-wide images.

## For Beginners

MultiDiffusion creates panoramic images wider than what the model
normally generates. It works by generating overlapping patches and blending them together
seamlessly, like taking multiple photos and stitching them into a panorama.

## How It Works

MultiDiffusion generates arbitrarily wide/tall images by running overlapping diffusion
passes and averaging the denoised results in the overlap regions. This produces seamless
panoramas without visible seam artifacts.

Reference: Bar-Tal et al., "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation", ICML 2023

