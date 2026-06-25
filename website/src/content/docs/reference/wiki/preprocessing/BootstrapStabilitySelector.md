---
title: "BootstrapStabilitySelector<T>"
description: "Bootstrap Stability-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Stability`

Bootstrap Stability-based Feature Selection.

## For Beginners

Bootstrap means taking many random samples
(with replacement) from your data. This method runs feature selection
on each sample and counts how often each feature is selected. Features
that are consistently selected are more reliable choices.

## How It Works

Uses bootstrap sampling to assess the stability of feature selection,
selecting features that are consistently chosen across multiple bootstrap
samples.

