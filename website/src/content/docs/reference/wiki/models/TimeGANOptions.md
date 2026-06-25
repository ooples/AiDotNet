---
title: "TimeGANOptions<T>"
description: "Configuration options for TimeGAN, a generative adversarial network designed specifically for generating realistic time-series data while preserving temporal dynamics."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TimeGAN, a generative adversarial network designed specifically
for generating realistic time-series data while preserving temporal dynamics.

## For Beginners

TimeGAN generates fake time-series data that looks realistic.

Regular GANs struggle with sequential data because they don't understand "time."
TimeGAN solves this with a clever multi-step training approach:

1. **Embedding training**: Learn to compress real data into a simpler form
2. **Supervised training**: Learn the rules of how data changes over time
3. **Joint training**: Train everything together — generator, discriminator,

embedder, recovery, and supervisor — all at once

Example:

## How It Works

TimeGAN combines autoencoding and adversarial training in a shared latent space.
It uses five components:

- **Embedder**: Maps real data to a latent embedding space
- **Recovery**: Reconstructs data from the latent space
- **Generator**: Produces synthetic latent embeddings
- **Supervisor**: Learns temporal dynamics in the latent space
- **Discriminator**: Distinguishes real from synthetic embeddings

Reference: "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DiscriminatorDropout` | Gets or sets the dropout rate for discriminator hidden layers. |
| `Epochs` | Gets or sets the total number of training epochs. |
| `HiddenDimension` | Gets or sets the hidden dimension for the RNN components. |
| `LearningRate` | Gets or sets the learning rate. |
| `NumFeatures` | Gets or sets the number of features per time step. |
| `NumLayers` | Gets or sets the number of RNN layers in each component. |
| `ReconstructionWeight` | Gets or sets the weight for the reconstruction loss. |
| `SequenceLength` | Gets or sets the length of each time-series sequence. |
| `SupervisedWeight` | Gets or sets the weight for the supervised loss in the generator objective. |

