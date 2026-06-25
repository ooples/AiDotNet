---
title: "TabSynOptions<T>"
description: "Configuration options for TabSyn, a state-of-the-art synthetic tabular data generator that combines a VAE with latent diffusion for high-quality generation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabSyn, a state-of-the-art synthetic tabular data generator
that combines a VAE with latent diffusion for high-quality generation.

## For Beginners

TabSyn is like TVAE + TabDDPM combined in a clever way:

1. First, a VAE learns to "compress" your data into a small summary (latent codes)
2. Then, a diffusion model learns the distribution of those compressed summaries
3. To generate new data: the diffusion model creates new summaries, and the VAE's

decoder converts them back to realistic data rows

This two-step approach often produces the highest quality synthetic data because:

- The VAE handles the complex mixed-type structure
- The diffusion model learns a simpler, continuous distribution in latent space

Example:

## How It Works

TabSyn operates in two phases:

1. **VAE pretraining**: Learns a compact latent representation of the tabular data
2. **Latent diffusion**: Trains a diffusion model in the VAE's latent space

Generation: Sample from the diffusion model in latent space, then decode with the VAE.

Reference: "TabSyn: Bridging the Gap" (Zhang et al., NeurIPS 2023)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `BetaEnd` | Gets or sets the beta schedule end value for diffusion. |
| `BetaStart` | Gets or sets the beta schedule start value for diffusion. |
| `DecoderDimensions` | Gets or sets the hidden layer sizes for the VAE decoder. |
| `DiffusionEpochs` | Gets or sets the number of epochs for diffusion model training. |
| `DiffusionLearningRate` | Gets or sets the learning rate for the diffusion model. |
| `DiffusionMLPDimensions` | Gets or sets the hidden layer sizes for the diffusion denoiser MLP. |
| `DiffusionSteps` | Gets or sets the number of diffusion timesteps. |
| `EncoderDimensions` | Gets or sets the hidden layer sizes for the VAE encoder. |
| `LatentDimension` | Gets or sets the dimension of the VAE latent space. |
| `TimestepEmbeddingDimension` | Gets or sets the dimension of the timestep embedding for the diffusion model. |
| `VAEEpochs` | Gets or sets the number of epochs for VAE pretraining. |
| `VAELearningRate` | Gets or sets the learning rate for the VAE. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column normalization. |

