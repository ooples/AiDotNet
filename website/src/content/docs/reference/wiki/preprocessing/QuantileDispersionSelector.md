---
title: "QuantileDispersionSelector<T>"
description: "Quantile Dispersion (QCD) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Quantile`

Quantile Dispersion (QCD) based Feature Selection.

## For Beginners

QCD measures spread relative to the central location,
similar to coefficient of variation but using quartiles instead of mean/std.
It's robust to outliers and works well when data contains extreme values.
Values range from 0 (no spread) to 1 (maximum relative spread).

## How It Works

Selects features based on the Quartile Coefficient of Dispersion,
a robust measure of relative variability: QCD = (Q3 - Q1) / (Q3 + Q1)

