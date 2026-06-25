---
title: "SelectKBest<T>"
description: "Selects the K best features based on a scoring function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Selects the K best features based on a scoring function.

## For Beginners

SelectKBest is the simplest approach - compute a score
for each feature independently and keep the top K. It's fast and works well when
features are truly independent, but may miss interactions between features.

## How It Works

A simple univariate feature selection that computes a score for each feature
and selects the K highest scoring features. The default scoring function
uses F-statistics for regression or classification.

