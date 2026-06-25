---
title: "RobustFeatureSelector<T>"
description: "Robust Feature Selection resistant to outliers and noise."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Robust`

Robust Feature Selection resistant to outliers and noise.

## For Beginners

Regular feature selection can be fooled by extreme
values (outliers) in the data. This method uses statistics that are "robust"

- they don't change much when a few data points are extreme. This gives

more reliable feature selection for messy real-world data.

## How It Works

Uses robust statistics (median, MAD, rank-based measures) to identify
important features while being resistant to outliers and noise in the data.

