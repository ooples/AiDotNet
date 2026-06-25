---
title: "MetaFDMixupOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-FDMixup: Feature-Distribution Mixup for cross-domain few-shot learning (Xu et al., CVPR 2021)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-FDMixup: Feature-Distribution Mixup for cross-domain
few-shot learning (Xu et al., CVPR 2021).

## How It Works

Meta-FDMixup improves cross-domain generalization by mixing feature distributions
(gradient signals) between tasks in a meta-batch. Instead of mixing raw inputs,
it mixes the gradient directions from different tasks, encouraging the meta-learner
to find an initialization robust across diverse task distributions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignmentWeight` | Weight for the feature distribution alignment loss. |
| `MixupAlpha` | Alpha parameter for the Beta distribution used to sample mixup coefficients. |
| `MixupProbability` | Probability of applying mixup to each task's gradient during inner loop. |

