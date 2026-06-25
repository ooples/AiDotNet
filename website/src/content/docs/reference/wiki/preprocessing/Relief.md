---
title: "Relief<T>"
description: "Relief algorithm for instance-based feature weighting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Relief`

Relief algorithm for instance-based feature weighting.

## For Beginners

For each sample, Relief looks at its nearest
neighbor from the same class (hit) and different class (miss). Good features
should be similar to hits and different from misses. This builds up a score
for each feature.

## How It Works

Relief estimates feature quality by sampling instances and comparing
to nearest hits (same class) and misses (different class). Features
differentiating classes get higher weights.

