---
title: "WinsorizedSelector<T>"
description: "Winsorized Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Robust`

Winsorized Feature Selection.

## For Beginners

Winsorizing means replacing the most extreme values
in your data with less extreme ones (typically at the 5th and 95th percentile).
This way, outliers don't disappear but are "capped" at reasonable values,
giving more robust feature importance measurements.

## How It Works

Uses Winsorization to replace extreme values with less extreme values before
computing feature importance, providing robust feature selection.

