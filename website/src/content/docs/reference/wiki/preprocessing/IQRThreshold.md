---
title: "IQRThreshold<T>"
description: "Interquartile Range (IQR) threshold feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Interquartile Range (IQR) threshold feature selection.

## For Beginners

IQR is the range between the 25th and 75th percentile.
It tells you how spread out the "typical" values are, ignoring extreme high or
low values. Features with small IQR have values clustered tightly together and
may not be useful for distinguishing samples.

## How It Works

Removes features whose IQR is below a threshold. IQR measures the spread of
the middle 50% of data and is robust to outliers.

