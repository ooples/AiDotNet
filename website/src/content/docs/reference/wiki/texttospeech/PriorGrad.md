---
title: "PriorGrad<T>"
description: "PriorGrad: adaptive diffusion vocoder that uses data-dependent prior (mel-conditioned noise) instead of isotropic Gaussian."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Vocoders`

PriorGrad: adaptive diffusion vocoder that uses data-dependent prior (mel-conditioned noise) instead of isotropic Gaussian.

## For Beginners

PriorGrad: adaptive diffusion vocoder that uses data-dependent prior (mel-conditioned noise) instead of isotropic Gaussian.. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior" (Lee et al., 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` | Converts mel to waveform using PriorGrad's data-dependent adaptive prior diffusion. |

