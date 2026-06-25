---
title: "TabDDPMOptions<T>"
description: "Configuration options for TabDDPM (Tabular Denoising Diffusion Probabilistic Model), a diffusion-based model for generating realistic synthetic tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabDDPM (Tabular Denoising Diffusion Probabilistic Model),
a diffusion-based model for generating realistic synthetic tabular data.

## For Beginners

TabDDPM works by gradually destroying data with noise, then learning
to reverse the process. Think of it like a restoration expert:

**Training (learning to restore):**

1. Take a real data row
2. Add a random amount of noise (more noise = more destroyed)
3. Tell the model "this was step t out of 1000" and ask it to predict the noise
4. The model learns to undo any amount of noise

**Generation (creating new data):**

1. Start with pure random noise
2. Ask the model to remove a tiny bit of noise (step 999 to 998)
3. Repeat 1000 times until you have a clean, realistic row

For numbers: regular Gaussian noise (like static on a TV)
For categories: noise means randomly changing the category toward "equally likely"

Example:

## How It Works

TabDDPM applies denoising diffusion models to tabular data with separate processes
for numerical and categorical features:

- **Gaussian diffusion** for continuous/numerical columns (adds Gaussian noise)
- **Multinomial diffusion** for categorical columns (transitions toward uniform distribution)
- A shared MLP denoiser with timestep embedding predicts the original data from noisy input

Reference: "TabDDPM: Modelling Tabular Data with Diffusion Models" (Kotelnikov et al., ICML 2023)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `BetaEnd` | Gets or sets the ending value of the beta schedule. |
| `BetaSchedule` | Gets or sets the beta (noise variance) schedule type. |
| `BetaStart` | Gets or sets the starting value of the beta schedule. |
| `DropoutRate` | Gets or sets the dropout rate for the denoiser MLP. |
| `Epochs` | Gets or sets the number of training epochs. |
| `LearningRate` | Gets or sets the learning rate for the optimizer. |
| `MLPDimensions` | Gets or sets the hidden layer sizes for the denoiser MLP. |
| `NumCategoricalDiffusionSteps` | Gets or sets the number of diffusion steps for the multinomial (categorical) diffusion process. |
| `NumTimesteps` | Gets or sets the number of diffusion timesteps. |
| `TimestepEmbeddingDimension` | Gets or sets the dimension of the timestep embedding. |

