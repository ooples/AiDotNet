---
title: "SelectPercentile<T>"
description: "Select features based on percentile of the highest scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection`

Select features based on percentile of the highest scores.

## For Beginners

Instead of saying "give me the best 5 features", you say
"give me the best 10% of features". This is useful when you don't know the total
number of features ahead of time or want a proportional selection.

## How It Works

SelectPercentile is similar to SelectKBest but instead of selecting a fixed number
of features, it selects features that score in the top percentile (e.g., top 10%).

