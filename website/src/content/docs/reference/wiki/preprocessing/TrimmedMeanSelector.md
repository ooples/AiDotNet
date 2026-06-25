---
title: "TrimmedMeanSelector<T>"
description: "Trimmed Mean-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Robust`

Trimmed Mean-based Feature Selection.

## For Beginners

When computing feature statistics, extreme values
(outliers) can skew results. Trimmed mean cuts off the highest and lowest
values before computing, giving you a more reliable picture of typical feature
behavior without outlier influence.

## How It Works

Uses trimmed statistics (removing extreme values) to compute robust feature
importance scores that are less affected by outliers.

