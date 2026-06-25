---
title: "PermutationImportance<T>"
description: "Permutation Importance feature selection by measuring score decrease when shuffling features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Permutation Importance feature selection by measuring score decrease when shuffling features.

## For Beginners

Imagine you have a trained model and want to know which features
it relies on. You shuffle one feature's values randomly (like shuffling a deck of cards)
and see how much worse the model performs. If performance drops a lot, that feature was
important. If shuffling makes no difference, the feature wasn't useful.

## How It Works

Permutation Importance measures the decrease in model performance when a single feature's
values are randomly shuffled, breaking the relationship between the feature and the target.
Features whose shuffling causes large performance drops are considered important.

