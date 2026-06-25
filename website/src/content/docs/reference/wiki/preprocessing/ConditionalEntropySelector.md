---
title: "ConditionalEntropySelector<T>"
description: "Conditional Entropy based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Entropy`

Conditional Entropy based Feature Selection.

## For Beginners

Conditional entropy measures how much uncertainty
remains about the target after knowing a feature's value. Features that reduce
uncertainty the most (lower conditional entropy) are most informative.

## How It Works

Selects features based on their ability to reduce uncertainty about the target,
measured by conditional entropy H(Y|X).

