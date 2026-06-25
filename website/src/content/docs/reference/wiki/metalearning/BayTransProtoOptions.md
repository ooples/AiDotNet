---
title: "BayTransProtoOptions<T, TInput, TOutput>"
description: "Configuration options for BayTransProto (Bayesian Transductive Prototypical Networks)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for BayTransProto (Bayesian Transductive Prototypical Networks).

## How It Works

BayTransProto extends prototypical networks with Bayesian parameter posteriors and
transductive refinement. The adapted parameters are sampled from a learned posterior,
and transductive steps use query predictions to iteratively refine the posterior mean.

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialLogVar` | Initial log-variance for posterior. |
| `KLWeight` | KL divergence weight for posterior regularization. |
| `NumPosteriorSamples` | Number of posterior samples during training. |
| `TransductiveLR` | Learning rate for transductive refinement. |
| `TransductiveSteps` | Transductive refinement steps using query predictions. |

