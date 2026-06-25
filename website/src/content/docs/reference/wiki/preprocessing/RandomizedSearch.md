---
title: "RandomizedSearch<T>"
description: "Randomized Search for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Randomized Search for feature selection.

## For Beginners

Instead of trying every combination (too slow) or
being systematic (might miss good solutions), this method randomly samples
feature combinations. It's like randomly picking lottery numbers - given enough
tries, you'll likely find something good.

## How It Works

Randomized Search evaluates random subsets of features to find good
combinations. It's faster than exhaustive search and can handle larger
feature spaces while still exploring diverse solutions.

