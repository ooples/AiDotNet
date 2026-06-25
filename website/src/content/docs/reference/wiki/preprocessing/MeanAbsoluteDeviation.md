---
title: "MeanAbsoluteDeviation<T>"
description: "Mean Absolute Deviation (MAD) for robust unsupervised feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Mean Absolute Deviation (MAD) for robust unsupervised feature selection.

## For Beginners

MAD tells you on average how far each value is from
the mean. Features with low MAD are nearly constant and probably not useful.
Unlike standard deviation, MAD isn't overly influenced by extreme outliers,
making it a safer choice for messy data.

## How It Works

Mean Absolute Deviation measures dispersion as the average distance from the mean.
Unlike variance/standard deviation, MAD is more robust to outliers because it
doesn't square the deviations.

