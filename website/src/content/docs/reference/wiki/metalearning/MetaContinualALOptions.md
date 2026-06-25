---
title: "MetaContinualALOptions<T, TInput, TOutput>"
description: "Configuration options for the MetaContinualAL (Meta-Continual Active Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the MetaContinualAL (Meta-Continual Active Learning) algorithm.

## How It Works

MetaContinualAL combines active learning sample selection with continual meta-learning.
It uses gradient-norm-based uncertainty estimation to identify the most informative
parameter dimensions, focusing adaptation on high-uncertainty regions while maintaining
a running calibration of uncertainty statistics.

## Properties

| Property | Summary |
|:-----|:--------|
| `AcquisitionFraction` | Fraction of parameters (by highest uncertainty) to focus adaptation on. |
| `ExplorationBonus` | Exploration bonus added to uncertain dimensions to encourage coverage. |
| `UncertaintyDecay` | EMA decay for running uncertainty statistics (mean/variance of gradient norms). |
| `UncertaintyWeight` | Weight on the uncertainty-guided gradient scaling. |

