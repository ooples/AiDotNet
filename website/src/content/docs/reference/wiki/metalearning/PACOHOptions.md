---
title: "PACOHOptions<T, TInput, TOutput>"
description: "Configuration options for PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters (Rothfuss et al., ICLR 2021)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters
(Rothfuss et al., ICLR 2021).

## How It Works

PACOH meta-learns a Gaussian prior N(μ, σ²I) over neural network parameters that
provides PAC-Bayesian generalization guarantees. The outer loop optimizes the prior
to minimize a PAC-Bayesian bound; the inner loop performs MAP estimation with
the learned prior as regularizer.

## Properties

| Property | Summary |
|:-----|:--------|
| `Delta` | Confidence parameter δ for the PAC-Bayesian bound (higher = tighter but more conservative). |
| `InitialLogVariance` | Initial log-variance of the prior distribution. |
| `KLCoefficient` | KL divergence coefficient in the PAC-Bayesian bound. |
| `NumPosteriorSamples` | Number of posterior samples for Monte Carlo estimation of expected loss. |

