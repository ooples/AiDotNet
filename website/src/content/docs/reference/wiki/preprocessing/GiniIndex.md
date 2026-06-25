---
title: "GiniIndex<T>"
description: "Gini Index feature selection for measuring impurity reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Gini Index feature selection for measuring impurity reduction.

## For Beginners

Imagine randomly guessing the class of an item. The Gini Index
measures how often you'd guess wrong. A feature that separates classes well will have
a low Gini Index (few wrong guesses). This is the same metric used by decision trees
like CART to choose splits.

## How It Works

The Gini Index measures the impurity of a split based on the probability of misclassifying
a randomly chosen element. Features that result in lower weighted Gini impurity after
splitting are considered more informative.

