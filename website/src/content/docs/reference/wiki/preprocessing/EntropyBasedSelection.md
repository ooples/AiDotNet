---
title: "EntropyBasedSelection<T>"
description: "Entropy-Based Feature Selection using discretized feature entropy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Entropy-Based Feature Selection using discretized feature entropy.

## For Beginners

Entropy measures how "mixed up" or unpredictable
a feature's values are. A feature that's always the same value has zero
entropy and tells you nothing useful. Features with higher entropy have
more variety and can better distinguish between different samples.

## How It Works

Measures the information content of each feature using Shannon entropy.
Features with higher entropy contain more information and discriminative power.
Features with very low entropy (constant or near-constant) are removed.

