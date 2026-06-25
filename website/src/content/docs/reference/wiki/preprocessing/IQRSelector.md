---
title: "IQRSelector<T>"
description: "Interquartile Range (IQR) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Range`

Interquartile Range (IQR) based Feature Selection.

## For Beginners

IQR is the difference between the 75th and 25th
percentiles. It's a robust measure of spread that isn't affected by outliers.
Features with higher IQR have more variation in their typical values.

## How It Works

Selects features based on their interquartile range, which measures
the spread of the middle 50% of data values.

