---
title: "LempelZivSelector<T>"
description: "Lempel-Ziv Complexity based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Complexity`

Lempel-Ziv Complexity based Feature Selection.

## For Beginners

Lempel-Ziv complexity counts how many unique
patterns appear when reading a sequence. It's related to compression -
sequences that compress well have low complexity (predictable patterns),
while complex sequences have many unique patterns and don't compress well.

## How It Works

Selects features based on their Lempel-Ziv complexity, which measures
the number of distinct patterns in a discretized sequence.

