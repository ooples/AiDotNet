---
title: "ChiSquare<T>"
description: "Chi-Square (χ²) test for feature selection in classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Classification`

Chi-Square (χ²) test for feature selection in classification.

## For Beginners

The Chi-Square test checks if the distribution of
a feature is different for different classes. If the values of a feature look
very different between classes, it gets a high score because that difference
can help predict class membership.

## How It Works

Chi-Square feature selection measures the dependence between features and
class labels using contingency tables. Features with high chi-square scores
have a significant statistical relationship with the target.

