---
title: "KernelPCASelector<T>"
description: "Kernel PCA-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Kernel`

Kernel PCA-based Feature Selection.

## For Beginners

Regular PCA finds linear patterns. Kernel PCA can
find non-linear patterns by implicitly mapping data to a higher dimension.
We select features that are most important in this richer representation,
capturing complex relationships that linear methods might miss.

## How It Works

Uses kernel PCA to map features to a higher-dimensional space and selects
features that contribute most to the principal components in kernel space.

