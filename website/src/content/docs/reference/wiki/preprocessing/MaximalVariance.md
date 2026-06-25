---
title: "MaximalVariance<T>"
description: "Maximal Variance feature selection for unsupervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Maximal Variance feature selection for unsupervised learning.

## For Beginners

If a feature has the same value for all samples,
it doesn't help distinguish between them. Features that vary a lot are more
informative. This method simply picks the features that change the most.

## How It Works

Maximal Variance selects features with the highest variance. Features with
higher variance contain more information and are more likely to be useful
for distinguishing between samples.

