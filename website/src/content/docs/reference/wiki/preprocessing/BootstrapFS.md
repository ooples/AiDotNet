---
title: "BootstrapFS<T>"
description: "Bootstrap Feature Selection for robust feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Stability`

Bootstrap Feature Selection for robust feature selection.

## For Beginners

Bootstrap is like asking the same question many
times with slightly different data each time. If a feature is selected most
of the time, it's probably genuinely important. If it only gets selected
sometimes, it might just be noise.

## How It Works

Bootstrap Feature Selection uses bootstrap sampling (sampling with replacement)
to estimate the stability of feature selection. Features that are consistently
selected across many bootstrap samples are more reliable.

