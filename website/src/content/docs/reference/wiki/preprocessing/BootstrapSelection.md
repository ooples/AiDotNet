---
title: "BootstrapSelection<T>"
description: "Bootstrap-based Feature Selection for robust feature importance estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Stability`

Bootstrap-based Feature Selection for robust feature importance estimation.

## For Beginners

Bootstrap is like asking the same question many
times to slightly different versions of your data (by random sampling with
replacement). This helps you understand which features are reliably important
and which ones might just be important by chance in one particular sample.

## How It Works

Uses bootstrap resampling (sampling with replacement) to estimate the
variability of feature importance scores. Features with consistently
high importance across bootstrap samples are selected.

