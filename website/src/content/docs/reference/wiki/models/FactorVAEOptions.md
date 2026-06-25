---
title: "FactorVAEOptions<T>"
description: "Configuration options for the FactorVAE model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the FactorVAE model.

## For Beginners

FactorVAE learns a compressed representation of market data.
You can decide how many hidden factors to discover and how strongly to enforce disentanglement.

## How It Works

FactorVAE is a variational autoencoder tailored to learning disentangled financial factors.
These options control the latent space size, factor count, and regularization strengths.

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta` | Gets or sets the beta coefficient for the VAE KL term. |
| `DropoutRate` | Gets or sets the dropout rate used for regularization. |
| `Gamma` | Gets or sets the gamma coefficient for the factor disentanglement penalty. |
| `HiddenDimension` | Gets or sets the width of hidden layers. |
| `LatentDimension` | Gets or sets the dimension of the latent space. |
| `NumAssets` | Gets or sets the number of assets covered by the model. |
| `NumFactors` | Gets or sets the number of latent factors to learn. |
| `NumFeatures` | Gets or sets the number of input features. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `SequenceLength` | Gets or sets the input sequence length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and throws if any value is invalid. |

