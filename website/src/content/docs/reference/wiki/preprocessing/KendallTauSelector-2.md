---
title: "KendallTauSelector<T>"
description: "Kendall's Tau Correlation Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

Kendall's Tau Correlation Feature Selection.

## For Beginners

Kendall's tau measures how well the ordering of one
variable matches the ordering of another. It counts pairs where both variables
agree on which item is "larger" vs pairs where they disagree. It's robust to
outliers because it only uses ranks, not actual values.

## How It Works

Selects features based on their Kendall's tau rank correlation with the target,
which is robust to outliers and non-linear monotonic relationships.

