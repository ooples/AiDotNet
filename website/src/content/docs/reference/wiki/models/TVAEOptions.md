---
title: "TVAEOptions<T>"
description: "Configuration options for TVAE (Tabular Variational Autoencoder), a VAE-based model for generating realistic synthetic tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TVAE (Tabular Variational Autoencoder), a VAE-based model
for generating realistic synthetic tabular data.

## For Beginners

TVAE learns to compress your data into a small "summary" (latent space)
and then reconstruct it back. Think of it like learning a recipe:

1. **Encoder**: Looks at a data row and writes a compact "recipe" (latent code)
2. **Decoder**: Reads the recipe and recreates the data row
3. **Training**: The model learns to write good recipes that can recreate realistic data
4. **Generation**: Sample random recipes from the latent space and decode them into new rows

TVAE is often faster to train than CTGAN and works well for moderate-sized datasets.

Example:

## How It Works

TVAE uses a variational autoencoder architecture adapted for tabular data:

- Same VGM normalization and one-hot encoding as CTGAN for preprocessing
- Encoder compresses data into a latent Gaussian distribution (mean, logvar)
- Decoder reconstructs data from sampled latent codes via reparameterization trick
- ELBO loss balances reconstruction quality with KL divergence regularization

Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DecoderDimensions` | Gets or sets the hidden layer sizes for the decoder network. |
| `EncoderDimensions` | Gets or sets the hidden layer sizes for the encoder network. |
| `Epochs` | Gets or sets the number of training epochs. |
| `LatentDimension` | Gets or sets the dimension of the latent space. |
| `LearningRate` | Gets or sets the learning rate for the optimizer. |
| `LossWeight` | Gets or sets the weight for reconstruction loss relative to KL divergence. |
| `VGMModes` | Gets or sets the number of Gaussian mixture components for VGM normalization. |

