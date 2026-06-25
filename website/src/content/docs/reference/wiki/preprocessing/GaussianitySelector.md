---
title: "GaussianitySelector<T>"
description: "Gaussianity (Normality) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Distribution`

Gaussianity (Normality) based Feature Selection.

## For Beginners

Many statistical methods assume data follows a
bell curve (normal distribution). This selector measures how "normal" each
feature is. You can select features that are most normal (for parametric methods)
or least normal (to find interesting non-standard patterns).

## How It Works

Selects features based on how closely their distributions match a Gaussian
(normal) distribution, using the Jarque-Bera test statistic.

