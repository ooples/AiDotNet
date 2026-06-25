---
title: "SelectPercentile<T>"
description: "Selects features based on a percentile of the highest scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Selects features based on a percentile of the highest scores.

## For Beginners

Instead of saying "keep the top 10 features," you say
"keep the top 25% of features." This is useful when you want a consistent reduction
ratio that scales with the number of features in your dataset.

## How It Works

Similar to SelectKBest but selects a percentage of features rather than a fixed
number. Useful when you want a consistent selection ratio across datasets of
different sizes.

