---
title: "SpikeAndSlabSelector<T>"
description: "Spike-and-Slab Feature Selection using Bayesian variable selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bayesian`

Spike-and-Slab Feature Selection using Bayesian variable selection.

## For Beginners

Imagine a spike-and-slab as two possibilities for
each feature: either it has no effect (spike at zero) or it has some effect
(spread out slab). The method estimates which possibility is more likely for
each feature. If the slab is more likely, the feature is important.

## How It Works

Implements the spike-and-slab prior model where each feature coefficient
has a mixture prior: a "spike" (concentrated at zero) and a "slab" (diffuse).
Features with high probability of being in the slab are selected.

