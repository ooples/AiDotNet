---
title: "ChiSquaredSelection<T>"
description: "Chi-Squared Feature Selection for categorical features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Chi-Squared Feature Selection for categorical features.

## For Beginners

The chi-squared test checks if two things
are related. If knowing the value of a feature helps predict the target,
they're related (high chi-squared). If the feature's value tells you
nothing about the target, chi-squared is low. This method keeps features
that are most related to what you're trying to predict.

## How It Works

Uses the chi-squared statistic to measure the dependency between each
feature and the target variable. Features with higher chi-squared values
have stronger associations with the target.

