---
title: "MADThreshold<T>"
description: "Median Absolute Deviation (MAD) threshold feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Median Absolute Deviation (MAD) threshold feature selection.

## For Beginners

Instead of using variance (which can be skewed by
extreme values), MAD looks at how far typical values are from the middle value.
This makes it better for data with outliers. Features with low MAD are nearly
constant and don't provide useful information.

## How It Works

Removes features whose Median Absolute Deviation is below a threshold. MAD
is a robust measure of spread that is less sensitive to outliers than variance.

