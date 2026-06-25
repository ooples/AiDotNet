---
title: "ExponentialDecaySelector<T>"
description: "Exponential Decay Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Streaming`

Exponential Decay Feature Selection.

## For Beginners

Unlike a sharp sliding window, exponential decay
smoothly reduces the importance of older samples. Recent data has the most
influence, but older data still contributes a little. This creates a smooth
transition as the data distribution changes.

## How It Works

Uses exponentially weighted statistics for feature selection, giving more
importance to recent samples while gradually forgetting older ones.

