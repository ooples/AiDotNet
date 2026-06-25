---
title: "DISR<T>"
description: "Double Input Symmetrical Relevance (DISR) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Double Input Symmetrical Relevance (DISR) for feature selection.

## For Beginners

DISR balances relevance and redundancy by looking at
how much two features together tell you about the target, normalized by their
total information content. It's like asking: "Of all the information in this
feature pair, what fraction is actually useful for prediction?"

## How It Works

DISR uses a symmetrical relevance measure that considers both feature-target and
feature-feature relationships. It maximizes a normalized joint relevance score
that accounts for redundancy in a principled way.

