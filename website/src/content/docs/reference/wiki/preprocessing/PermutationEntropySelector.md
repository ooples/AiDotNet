---
title: "PermutationEntropySelector<T>"
description: "Permutation Entropy based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Entropy`

Permutation Entropy based Feature Selection.

## For Beginners

Permutation entropy looks at the order of values
in small windows. It counts how many different orderings appear and how often.
High permutation entropy means complex, unpredictable sequences; low entropy
means regular, predictable patterns like trends.

## How It Works

Selects features based on permutation entropy, which measures the complexity
of time series or sequential data based on ordinal patterns.

