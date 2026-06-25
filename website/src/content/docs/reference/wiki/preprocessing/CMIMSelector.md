---
title: "CMIMSelector<T>"
description: "Conditional Mutual Information Maximization (CMIM) Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Multivariate`

Conditional Mutual Information Maximization (CMIM) Feature Selection.

## For Beginners

CMIM asks "what new information does this feature
give me that I don't already have?" It selects features that add something new,
conditioned on what you've already picked. This avoids redundant features more
rigorously than simple redundancy penalties.

## How It Works

Selects features by maximizing mutual information with the target conditioned
on already selected features, ensuring each new feature adds unique information.

