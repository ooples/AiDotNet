---
title: "MADBasedSelector<T>"
description: "Median Absolute Deviation (MAD) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Robust`

Median Absolute Deviation (MAD) based Feature Selection.

## For Beginners

Regular variance can be heavily influenced by
outliers (extreme values). MAD uses the median instead of the mean, making
it much more robust. Features with higher MAD have more genuine variability,
not just variability caused by a few extreme values.

## How It Works

Uses Median Absolute Deviation as a robust measure of feature spread that is
less sensitive to outliers than variance-based methods.

