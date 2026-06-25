---
title: "AutoDiffTabOptions<T>"
description: "Configuration options for AutoDiff-Tab, an automated diffusion model for tabular data that searches over diffusion configurations and noise schedules to find optimal settings."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for AutoDiff-Tab, an automated diffusion model for tabular data
that searches over diffusion configurations and noise schedules to find optimal settings.

## For Beginners

AutoDiff-Tab is like TabDDPM with automatic tuning:

Instead of manually setting "how noisy" and "how many steps" the diffusion process uses,
AutoDiff-Tab tries different settings and picks the best one automatically.

It explores:

1. Different amounts of noise (noise schedule)
2. Different numbers of denoising steps
3. Different neural network sizes

Example:

## How It Works

AutoDiff-Tab automates the design of diffusion models for tabular data by searching over:

- **Number of diffusion timesteps**: How many denoising steps
- **Beta schedule**: Linear, cosine, or learned noise schedule
- **MLP architecture**: Depth, width, and dropout of the denoiser
- **Noise schedule parameters**: Start/end beta values

Reference: "Automated Diffusion Models for Tabular Data" (2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `BetaEnd` | Gets or sets the maximum beta value for the noise schedule. |
| `BetaSchedule` | Gets or sets the type of noise schedule. |
| `BetaStart` | Gets or sets the minimum beta value for the noise schedule. |
| `DropoutRate` | Gets or sets the dropout rate for the denoiser. |
| `Epochs` | Gets or sets the number of training epochs. |
| `LearningRate` | Gets or sets the learning rate. |
| `MLPDimensions` | Gets or sets the hidden layer dimensions for the denoiser MLP. |
| `MaxTimesteps` | Gets or sets the maximum number of diffusion timesteps to search over. |
| `SearchTrials` | Gets or sets the number of search trials to find optimal diffusion configuration. |
| `TimestepEmbeddingDimension` | Gets or sets the dimension of timestep embeddings. |
| `TrialEpochs` | Gets or sets the number of epochs for each search trial (reduced for speed). |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column transformation. |

