---
title: "MaximalInformation<T>"
description: "Maximal Information feature selection based on entropy for unsupervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Maximal Information feature selection based on entropy for unsupervised learning.

## For Beginners

Entropy measures how "surprising" or "uncertain"
a feature's values are. A feature that's always the same is boring (low entropy).
A feature with lots of different values is informative (high entropy). This method
selects the most informative features.

## How It Works

Maximal Information selects features with highest entropy (information content).
Features with higher entropy contain more information and can better capture
the underlying structure of the data.

