---
title: "GiniImpuritySelector<T>"
description: "Gini Impurity based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Gain`

Gini Impurity based Feature Selection.

## For Beginners

Gini impurity measures how "mixed" a set of labels is.
Pure sets (all same class) have Gini = 0. Features that best split data into
purer groups have higher Gini importance. This is fast to compute and widely
used in tree-based methods like Random Forest.

## How It Works

Selects features based on Gini impurity reduction, the same criterion
used by CART decision trees to choose split features.

