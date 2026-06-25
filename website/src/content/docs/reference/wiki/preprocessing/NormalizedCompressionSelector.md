---
title: "NormalizedCompressionSelector<T>"
description: "Normalized Compression Distance based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Compression`

Normalized Compression Distance based Feature Selection.

## For Beginners

This measures how much "shared information" exists
between a feature and the target. Features that compress well together with
the target share more mutual information.

## How It Works

Selects features based on their normalized compression distance to the target,
approximating Kolmogorov complexity for feature relevance.

