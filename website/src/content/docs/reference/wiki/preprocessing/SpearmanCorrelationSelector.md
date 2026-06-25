---
title: "SpearmanCorrelationSelector<T>"
description: "Spearman Rank Correlation based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Similarity`

Spearman Rank Correlation based Feature Selection.

## For Beginners

Spearman correlation works on ranks rather than raw values.
It detects if one variable consistently increases when another increases, even if
the relationship isn't a straight line. It's more robust to outliers than Pearson.

## How It Works

Selects features based on their Spearman rank correlation with the target,
measuring monotonic relationships (not necessarily linear).

