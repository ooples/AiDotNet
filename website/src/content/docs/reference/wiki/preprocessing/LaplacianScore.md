---
title: "LaplacianScore<T>"
description: "Laplacian Score for unsupervised feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral`

Laplacian Score for unsupervised feature selection.

## For Beginners

The Laplacian Score measures how smoothly a feature
varies across similar data points. A good feature should have similar values
for nearby points (like neighbors having similar house prices). Features that
jump around randomly between neighbors score poorly.

## How It Works

Laplacian Score evaluates features based on their ability to preserve locality
in the data. Features with low Laplacian Score preserve local structure well,
meaning similar samples have similar feature values.

