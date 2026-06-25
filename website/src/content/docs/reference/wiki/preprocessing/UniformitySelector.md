---
title: "UniformitySelector<T>"
description: "Uniformity based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Distribution`

Uniformity based Feature Selection.

## For Beginners

A uniform distribution has equal probability for
all values in a range. This selector measures how close each feature is to
being uniformly distributed. High uniformity means values are evenly spread out,
while low uniformity means some values are much more common than others.

## How It Works

Selects features based on how uniformly distributed their values are,
using entropy as a measure of uniformity.

