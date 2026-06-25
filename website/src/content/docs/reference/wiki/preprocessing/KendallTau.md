---
title: "KendallTau<T>"
description: "Kendall Tau Correlation for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation`

Kendall Tau Correlation for feature selection.

## For Beginners

Kendall Tau looks at every pair of data points and
asks: "Do they agree on the ranking?" If both feature and target say A > B,
that's concordant. If they disagree, that's discordant. More concordant pairs
mean stronger positive correlation.

## How It Works

Kendall Tau measures the ordinal association between features and target by
counting concordant and discordant pairs. It's robust to outliers and doesn't
assume a specific distribution.

