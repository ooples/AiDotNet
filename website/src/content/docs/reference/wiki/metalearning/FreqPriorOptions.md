---
title: "FreqPriorOptions<T, TInput, TOutput>"
description: "Configuration options for FreqPrior: Frequency-based prior for cross-domain few-shot learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for FreqPrior: Frequency-based prior for cross-domain few-shot learning.

## How It Works

FreqPrior decomposes the parameter space into frequency bands using a discrete cosine-like
transform. Low-frequency components capture domain-invariant structure and are strongly
regularized toward the meta-learned prior, while high-frequency components are allowed to
vary freely for task-specific adaptation. This frequency-based prior encourages learning
smooth, transferable representations.

## Properties

| Property | Summary |
|:-----|:--------|
| `FrequencyPenaltyScale` | Scale factor for the frequency penalty term applied to low-frequency gradient variance. |
| `HighFreqRegWeight` | Regularization strength for high-frequency components (allows more variation). |
| `LowFreqFraction` | Fraction of parameters considered "low frequency" (strongly regularized). |
| `LowFreqRegWeight` | Regularization strength for low-frequency components toward the meta-prior. |

