---
title: "NPOptions<T, TInput, TOutput>"
description: "Configuration options for Neural Process (NP) (Garnelo et al., 2018)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Neural Process (NP) (Garnelo et al., 2018).

## How It Works

NP extends CNP with a latent variable z that captures global uncertainty about the function.
The ELBO objective balances reconstruction accuracy with KL divergence regularization.

## Properties

| Property | Summary |
|:-----|:--------|
| `KLWeight` | Gets or sets the KL divergence weight in the ELBO. |
| `LatentDim` | Gets or sets the latent variable dimensionality. |
| `NumLatentSamples` | Gets or sets the number of latent samples for prediction. |
| `RepresentationDim` | Gets or sets the representation dimensionality. |

