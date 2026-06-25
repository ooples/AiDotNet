---
title: "InterquartileRange<T>"
description: "Interquartile Range (IQR) for robust unsupervised feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Interquartile Range (IQR) for robust unsupervised feature selection.

## For Beginners

IQR focuses on the "typical" range of values, ignoring
the extreme high and low values. A feature with small IQR has most values bunched
together and may not be discriminative. IQR is great for data with outliers because
it completely ignores them.

## How It Works

The interquartile range is the difference between the 75th and 25th percentiles.
It measures the spread of the middle 50% of data and is extremely robust to outliers.

