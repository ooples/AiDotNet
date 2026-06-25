---
title: "CSDIOptions<T>"
description: "Configuration options for CSDI (Conditional Score-based Diffusion model for Imputation)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for CSDI (Conditional Score-based Diffusion model for Imputation).

## For Beginners

CSDI solves a critical problem in real-world data:
missing values. Instead of simple interpolation, it generates plausible values
that are consistent with the observed data.

**The Key Insight:**
Unlike TimeGrad which forecasts future values, CSDI focuses on imputation:
filling in missing values WITHIN the observed time series. It conditions on
what you DO know to infer what you DON'T know.

**How CSDI Works:**

1. **Conditional Masking:** Identify which values are observed vs missing
2. **Score Matching:** Learn the gradient of log probability (the "score")
3. **Reverse Diffusion:** Start from noise, gradually denoise conditioned on observed values
4. **Imputation:** Generate multiple samples for uncertainty quantification

**Architecture:**

- Transformer-based score network with self-attention
- Temporal and feature embeddings for position encoding
- Conditional U-Net style residual blocks
- Side information integration for covariates

**Key Benefits:**

- Handles arbitrary missing patterns (not just regular gaps)
- Provides uncertainty estimates for imputed values
- Can incorporate side information (covariates)
- State-of-the-art imputation quality

## How It Works

CSDI is a probabilistic model for time series imputation that uses score-based
diffusion to fill in missing values with well-calibrated uncertainty estimates.

**Reference:** Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", 2021.
https://arxiv.org/abs/2107.03502

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CSDIOptions` | Initializes a new instance of the `CSDIOptions` class with default values. |
| `CSDIOptions(CSDIOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the ending noise level (beta_T). |
| `BetaSchedule` | Gets or sets the noise schedule type. |
| `BetaStart` | Gets or sets the starting noise level (beta_1). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `FeatureEmbeddingDim` | Gets or sets the dimension of feature embeddings. |
| `HiddenDimension` | Gets or sets the hidden dimension for the score network. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps. |
| `NumFeatures` | Gets or sets the number of features (variables). |
| `NumHeads` | Gets or sets the number of attention heads in the transformer layers. |
| `NumResidualLayers` | Gets or sets the number of residual layers in the score network. |
| `NumSamples` | Gets or sets the number of samples to generate for uncertainty estimation. |
| `SequenceLength` | Gets or sets the sequence length (time steps). |
| `SideInfoDim` | Gets or sets the dimension of side information features. |
| `TimeEmbeddingDim` | Gets or sets the dimension of time step embeddings. |
| `UseAttention` | Gets or sets whether to use self-attention in the score network. |
| `UseSideInfo` | Gets or sets whether to use side information (covariates). |

