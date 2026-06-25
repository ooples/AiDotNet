---
title: "MICorrelationHybrid<T>"
description: "Mutual Information and Correlation Hybrid Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Mutual Information and Correlation Hybrid Feature Selection.

## For Beginners

Some features have linear relationships
(more X means more Y), while others have non-linear patterns (X affects
Y in complex ways). This method uses correlation to find linear patterns
and mutual information to find any pattern, then combines both scores.

## How It Works

Combines mutual information (captures non-linear relationships) with
Pearson correlation (captures linear relationships) to identify features
that are important under both perspectives.

