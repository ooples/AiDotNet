---
title: "DOVE<T>"
description: "DOVE: harnessing large-scale video diffusion priors for general video restoration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

DOVE: harnessing large-scale video diffusion priors for general video restoration.

## For Beginners

DOVE repurposes a video generation AI to restore degraded video.
Think of it like asking an artist who knows what clean video looks like to paint a
restored version of your noisy/blurry footage, rather than trying to mathematically
reverse the damage.

**Usage:**

## How It Works

DOVE (Chen et al., 2025) uses pretrained video diffusion models as powerful restoration priors:

- Degradation estimation: analyzes input video to determine the type and severity of degradation
- Guided generation: conditions a pretrained Stable Video Diffusion backbone on the

degradation estimate, generating clean video through controlled denoising

- Video diffusion prior: the SVD backbone's knowledge of natural video dynamics provides

inherent temporal consistency without explicit flow or alignment modules

Unlike discriminative approaches (BasicVSR, EDVR), DOVE is generative: it synthesizes
plausible clean video conditioned on the degraded input.

**Reference:** "DOVE: Harnessing Large-Scale Video Diffusion Priors for General Video
Restoration" (Chen et al., 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DOVE(NeuralNetworkArchitecture<>,DOVEOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DOVE model in native training mode. |
| `DOVE(NeuralNetworkArchitecture<>,String,DOVEOptions)` | Creates a DOVE model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

