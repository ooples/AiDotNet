---
title: "MutualInformationRegression<T>"
description: "Mutual Information for regression feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Mutual Information for regression feature selection.

## For Beginners

For regression problems where the target is a number
(not a category), mutual information still works by grouping both the feature
and target into bins. High MI means knowing the feature helps predict the target
value, capturing both linear and nonlinear relationships.

## How It Works

Measures the mutual information between each feature and a continuous target.
Uses binning to estimate the probability distributions for computing MI.

