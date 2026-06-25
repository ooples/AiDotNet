---
title: "PermutationImportance<T>"
description: "Permutation Importance for model-agnostic feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic`

Permutation Importance for model-agnostic feature selection.

## For Beginners

This method works with any model. It asks: "What happens
if I scramble this feature's values?" If the model gets much worse, that feature
was important. If the model doesn't care, the feature wasn't useful. It's like
testing each ingredient by removing it from a recipe.

## How It Works

Permutation Importance measures feature importance by randomly shuffling each feature
and observing how much model performance degrades. Features that cause large
performance drops when shuffled are important.

