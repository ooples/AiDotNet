---
title: "GiniImpuritySelector<T>"
description: "Gini Impurity based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Classification`

Gini Impurity based Feature Selection.

## For Beginners

Gini impurity measures how mixed up the classes are.
A pure group (all same class) has impurity 0. This selector keeps features that
best reduce impurity when used to split the data - the same criterion decision
trees use.

## How It Works

Selects features based on their ability to reduce Gini impurity when used
for splitting data, similar to decision tree feature importance.

