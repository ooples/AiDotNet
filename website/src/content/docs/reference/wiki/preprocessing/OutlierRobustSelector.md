---
title: "OutlierRobustSelector<T>"
description: "Outlier Robust Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Robustness`

Outlier Robust Feature Selection.

## For Beginners

Some features are heavily influenced by extreme values
(outliers). This selector finds features whose relationship with the target is
consistent whether outliers are present or removed.

## How It Works

Selects features that maintain their predictive relationship with the target
even when outliers are present, using robust statistics.

