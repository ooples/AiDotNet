---
title: "FusedLasso<T>"
description: "Fused Lasso (Total Variation) for feature selection with smoothness constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sparse`

Fused Lasso (Total Variation) for feature selection with smoothness constraints.

## For Beginners

When features are ordered (like wavelengths or time points),
you often expect nearby features to behave similarly. Fused Lasso encourages smooth
coefficient patterns while still allowing for sparse solutions where coefficients
jump abruptly only at important boundaries.

## How It Works

Fused Lasso adds a penalty on the differences between consecutive coefficients,
encouraging nearby features to have similar coefficients. This is useful when
features have a natural ordering (e.g., spectral data, time series).

