---
title: "FlexPACBayesOptions<T, TInput, TOutput>"
description: "Configuration options for Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning with data-dependent prior construction."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning
with data-dependent prior construction.

## How It Works

Flex-PAC-Bayes extends PAC-Bayesian meta-learning by constructing the prior from a
fraction of the support data ("prior data") and computing the PAC-Bayesian bound
on the remaining data ("bound data"). This data-dependent prior construction yields
tighter generalization bounds. The "flex" parameter (λ) interpolates between
standard PAC-Bayes (λ=1) and pure empirical risk minimization (λ→0).

**Key bound:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Delta` | Confidence parameter δ for the PAC-Bayesian bound. |
| `FlexParameter` | Flex parameter λ controlling the trade-off between bound tightness and regularization. |
| `InitialLogVariance` | Initial log-variance for the posterior distribution. |
| `KLCoefficient` | KL divergence coefficient in the PAC-Bayesian bound. |
| `PriorDataFraction` | Fraction of support data used to construct the data-dependent prior (0 < f < 1). |

