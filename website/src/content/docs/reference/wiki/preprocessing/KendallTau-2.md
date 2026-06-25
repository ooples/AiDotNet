---
title: "KendallTau<T>"
description: "Kendall Tau correlation for feature selection based on concordant pairs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Kendall Tau correlation for feature selection based on concordant pairs.

## For Beginners

Kendall Tau looks at every pair of data points and asks:
when the feature goes up, does the target also go up? If most pairs agree (concordant),
the correlation is high. Unlike Pearson, it doesn't assume linear relationships and
is very robust to outliers. It's especially good for ordinal or ranked data.

## How It Works

Kendall's Tau measures the ordinal association between features and target by
comparing pairs of observations. It counts concordant pairs (both values increase
or both decrease) versus discordant pairs (one increases while the other decreases).

