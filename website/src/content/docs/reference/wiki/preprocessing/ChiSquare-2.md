---
title: "ChiSquare<T>"
description: "Chi-Square test for feature selection in classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Chi-Square test for feature selection in classification problems.

## For Beginners

Chi-Square asks: "Is there a relationship between this
feature and the target class, or could the observed pattern be due to chance?"
High chi-square values indicate a strong association. It works best with categorical
or discretized features.

## How It Works

The Chi-Square test measures the independence between each feature and the target
class. Features that are strongly associated with the target (high chi-square statistic)
are considered more relevant for classification.

