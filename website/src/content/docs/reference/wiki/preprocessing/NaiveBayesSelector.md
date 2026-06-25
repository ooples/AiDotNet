---
title: "NaiveBayesSelector<T>"
description: "Naive Bayes based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Classification`

Naive Bayes based Feature Selection.

## For Beginners

Naive Bayes assumes features are independent given
the class. This selector measures how much each feature helps distinguish
between classes by looking at how different the feature distributions are
for each class.

## How It Works

Selects features based on their discriminative power using Naive Bayes
class-conditional probability estimates.

