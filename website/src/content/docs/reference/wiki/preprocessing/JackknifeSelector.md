---
title: "JackknifeSelector<T>"
description: "Jackknife based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Stability`

Jackknife based Feature Selection.

## For Beginners

Jackknife is a resampling method that removes one
data point at a time and recalculates statistics. Features with consistent
importance scores across all jackknife samples are more reliable.

## How It Works

Selects features based on their stability under jackknife resampling,
which systematically leaves out one sample at a time.

