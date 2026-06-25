---
title: "MADSelector<T>"
description: "Median Absolute Deviation (MAD) based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Median Absolute Deviation (MAD) based feature selection.

## For Beginners

Standard variance can be thrown off by outliers.
MAD uses the median instead of mean, making it more robust. Features with
higher MAD have more consistent spread in their values.

## How It Works

Uses MAD as a robust measure of variability. Unlike variance, MAD is
resistant to outliers, making it better for selecting features in
datasets with extreme values.

