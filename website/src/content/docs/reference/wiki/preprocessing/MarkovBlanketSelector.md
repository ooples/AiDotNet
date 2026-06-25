---
title: "MarkovBlanketSelector<T>"
description: "Markov Blanket Feature Selection for finding causal features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Causal`

Markov Blanket Feature Selection for finding causal features.

## For Beginners

Imagine the target variable has a "blanket" of
features around it that shields it from all other features. If you know
the blanket features, knowing any other feature doesn't help predict the
target. This method finds that protective blanket of features.

## How It Works

Identifies the Markov blanket of the target variable - the minimal set of
features that makes the target conditionally independent of all other
features. These features are the most relevant for prediction.

