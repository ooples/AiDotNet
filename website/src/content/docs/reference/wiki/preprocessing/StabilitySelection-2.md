---
title: "StabilitySelection<T>"
description: "Stability Selection for robust feature selection with FDR control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Stability Selection for robust feature selection with FDR control.

## For Beginners

By running selection on many random subsets
of data, we find features that are "stably" selected regardless of
which samples we use. These are more likely to be truly important.

## How It Works

Runs a base feature selector on many bootstrap samples and selects
features that are consistently chosen across subsamples. Provides
error rate control.

