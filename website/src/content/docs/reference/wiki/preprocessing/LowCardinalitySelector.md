---
title: "LowCardinalitySelector<T>"
description: "Low Cardinality Feature Removal."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Low Cardinality Feature Removal.

## For Beginners

Cardinality is the number of unique values a
feature has. A feature with only 1 or 2 unique values doesn't vary much
and may not be useful for distinguishing between samples. This selector
removes such low-variety features.

## How It Works

Removes features with very low cardinality (few unique values) as they
may not provide enough discriminative power.

