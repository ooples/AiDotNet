---
title: "ConditionalEntropySelector<T>"
description: "Conditional Entropy-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Information`

Conditional Entropy-based Feature Selection.

## For Beginners

Conditional entropy measures the uncertainty
remaining about the target after knowing a feature's value. Features that
reduce uncertainty the most (lower conditional entropy) are more useful.

## How It Works

Selects features that minimize the conditional entropy of the target
given the feature, maximizing the information about the target.

