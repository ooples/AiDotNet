---
title: "KendallCorrelation<T>"
description: "Kendall Tau Correlation-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation`

Kendall Tau Correlation-based feature selection.

## For Beginners

Kendall's Tau looks at pairs of data points and asks:
"when one goes up, does the other also go up?" It counts how many pairs agree
versus disagree. It's particularly good when you have a small dataset or many
tied values, and gives a more intuitive probability interpretation.

## How It Works

Kendall Tau measures the ordinal association between features and target
by counting concordant and discordant pairs. It's more robust than Spearman
for small samples and handles ties well.

