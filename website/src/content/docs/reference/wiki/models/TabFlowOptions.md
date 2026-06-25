---
title: "TabFlowOptions<T>"
description: "Configuration options for TabFlow, a flow matching model for generating synthetic tabular data using continuous normalizing flows with optimal transport paths."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabFlow, a flow matching model for generating synthetic tabular data
using continuous normalizing flows with optimal transport paths.

## For Beginners

TabFlow learns to gradually transform random noise into realistic data
through a smooth, deterministic path (no randomness during generation):

1. **Training**: Learn a velocity field that moves noise toward data along straight paths
2. **Generation**: Start from noise and follow the velocity field using an ODE solver

Advantages over diffusion models:

- Faster generation (fewer steps needed since paths are straighter)
- Deterministic sampling (same noise → same output)
- Often higher quality for tabular data

Example:

## How It Works

TabFlow uses flow matching (a continuous normalizing flow approach) to learn a deterministic
mapping from noise to data. Unlike diffusion models that use stochastic differential equations,
TabFlow uses ordinary differential equations (ODEs) with optimal transport conditional paths.

Reference: "Flow Matching for Tabular Data" (2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DropoutRate` | Gets or sets the dropout rate for the MLP. |
| `Epochs` | Gets or sets the number of training epochs. |
| `LearningRate` | Gets or sets the learning rate. |
| `MLPDimensions` | Gets or sets the hidden layer sizes for the velocity field MLP. |
| `NumSteps` | Gets or sets the number of ODE solver steps for generation. |
| `Sigma` | Gets or sets the sigma for the optimal transport conditional flow. |
| `Solver` | Gets or sets the ODE solver type. |
| `TimeEmbeddingDimension` | Gets or sets the dimension of the time embedding. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous columns. |

