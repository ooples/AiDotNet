---
title: "MissingValueSelector<T>"
description: "Missing Value-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Missing Value-based Feature Selection.

## For Beginners

Features with lots of missing data can't be
reliably used for predictions. This selector removes features where too
much data is missing, keeping only the ones with enough information.

## How It Works

Removes features that have too many missing values (zeros, NaN, or
below-threshold unique values), keeping only features with sufficient data.

