---
title: "ChiSquareTest<T>"
description: "Chi-square test for feature selection in classification with categorical features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Chi-square test for feature selection in classification with categorical features.

## For Beginners

Chi-square tests whether a feature's values are
distributed differently across classes. If feature values are independent of
the class (evenly distributed), they can't help predict the class. High
chi-square scores indicate strong association with the target.

## How It Works

Uses the chi-square test of independence to measure the association between
features and target classes. Features with high chi-square values show strong
dependence on the target and are selected.

