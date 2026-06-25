---
title: "GiniIndex<T>"
description: "Gini Index for feature selection based on impurity reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Gini Index for feature selection based on impurity reduction.

## For Beginners

Gini measures how mixed a group is. A group
with all same-class items has Gini=0 (pure). A 50/50 split has Gini=0.5.
Good features create pure groups when you split by them.

## How It Works

The Gini Index measures impurity in classification. Features that split
data into purer groups (lower Gini) are more informative for prediction.

