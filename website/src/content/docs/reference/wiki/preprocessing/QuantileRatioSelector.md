---
title: "QuantileRatioSelector<T>"
description: "Quantile Ratio based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Quantile`

Quantile Ratio based Feature Selection.

## For Beginners

Quantile ratio compares values at different
percentiles. For example, the 90th/10th percentile ratio shows how much
larger the high values are compared to low values. This can detect features
with skewed distributions or outliers.

## How It Works

Selects features based on the ratio between two quantiles, measuring
the relative spread at different parts of the distribution.

