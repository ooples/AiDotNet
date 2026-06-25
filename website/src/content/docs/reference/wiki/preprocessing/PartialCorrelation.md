---
title: "PartialCorrelation<T>"
description: "Partial Correlation for feature selection controlling for other variables."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation`

Partial Correlation for feature selection controlling for other variables.

## For Beginners

Regular correlation can be misleading if two
variables are both caused by a third (confounder). Partial correlation
"holds constant" other variables to find the true direct relationship.

## How It Works

Partial correlation measures the relationship between two variables while
controlling for (removing the effect of) other variables. Helps identify
direct relationships vs those mediated by confounders.

