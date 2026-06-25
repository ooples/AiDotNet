---
title: "ErosionDilationSelector<T>"
description: "Erosion/Dilation based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Morphological`

Erosion/Dilation based Feature Selection.

## For Beginners

Morphological operations like erosion (shrinking)
and dilation (expanding) reveal structural properties of data. Features that
change significantly under these operations may capture important local patterns.

## How It Works

Selects features based on morphological properties, measuring how feature values
respond to erosion and dilation operations (local minimum/maximum filtering).

