---
title: "SelectKBest<T>"
description: "Select K Best features using a pluggable scoring function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection`

Select K Best features using a pluggable scoring function.

## For Beginners

This is a versatile "pick the best k features" tool.
You provide a scoring function (like correlation, F-statistic, or mutual information),
and it simply picks the top k scoring features. It's scikit-learn compatible in concept.

## How It Works

SelectKBest is a simple feature selector that computes a score for each feature
using a provided scoring function and selects the k features with highest scores.

