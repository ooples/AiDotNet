---
title: "FRegression<T>"
description: "F-Regression for feature selection based on linear relationship with continuous target."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

F-Regression for feature selection based on linear relationship with continuous target.

## For Beginners

For regression problems (predicting continuous values),
we need to know which features have a linear relationship with what we're predicting.
F-regression fits a simple line between each feature and the target, then measures
how well that line explains the variation in the target.

## How It Works

Computes the F-statistic for a univariate linear regression between each feature and
the continuous target. Features with higher F-statistics have stronger linear
relationships with the target.

