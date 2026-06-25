---
title: "ApproximateEntropySelector<T>"
description: "Approximate Entropy (ApEn) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Entropy`

Approximate Entropy (ApEn) based Feature Selection.

## For Beginners

Approximate entropy measures how likely it is
that similar patterns in a sequence remain similar when extended by one point.
Low ApEn means the sequence is predictable (patterns repeat); high ApEn means
it's complex and hard to predict. Often used for physiological signals.

## How It Works

Selects features based on their approximate entropy, which quantifies
the unpredictability and complexity of time series data.

