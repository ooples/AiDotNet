---
title: "FinDiffOptions<T>"
description: "Configuration options for FinDiff, a diffusion model specialized for generating realistic financial tabular data with temporal correlation preservation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FinDiff, a diffusion model specialized for generating
realistic financial tabular data with temporal correlation preservation.

## For Beginners

FinDiff generates realistic financial data (stocks, portfolios, etc.)
by understanding that financial data has special properties:

1. Values change gradually over time (temporal correlation)
2. Some values must always be positive (stock prices)
3. Volatility clusters (periods of high/low market turbulence)

Example:

## How It Works

FinDiff extends standard diffusion models with financial domain knowledge:

- **Temporal correlation loss**: Preserves autocorrelation and cross-correlation
- **Volatility-aware noise**: Adapts noise schedule to financial volatility patterns
- **Financial constraints**: Enforces domain rules (positive prices, valid ranges)

Reference: "Diffusion Models for Financial Tabular Data" (2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `BetaEnd` | Gets or sets the maximum beta for the noise schedule. |
| `BetaStart` | Gets or sets the minimum beta for the noise schedule. |
| `EnforcePositive` | Gets or sets whether to enforce positive values in generated data. |
| `Epochs` | Gets or sets the number of training epochs. |
| `LearningRate` | Gets or sets the learning rate. |
| `MLPDimensions` | Gets or sets the hidden layer dimensions for the denoiser MLP. |
| `NumTimesteps` | Gets or sets the number of diffusion timesteps. |
| `TemporalWeight` | Gets or sets the weight for the temporal correlation loss. |
| `TimestepEmbeddingDimension` | Gets or sets the timestep embedding dimension. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column transformation. |

