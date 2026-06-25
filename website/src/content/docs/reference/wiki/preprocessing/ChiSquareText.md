---
title: "ChiSquareText<T>"
description: "Chi-Square test for text classification feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Text`

Chi-Square test for text classification feature selection.

## For Beginners

This test finds words that appear significantly
more often in one class than expected by chance. These discriminative words
are the best features for text classification.

## How It Works

Applies chi-square test to measure the association between terms and
document classes. Terms with high chi-square values are more discriminative.

