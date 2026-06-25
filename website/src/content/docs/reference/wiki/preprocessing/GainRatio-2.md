---
title: "GainRatio<T>"
description: "Gain Ratio for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Gain Ratio for feature selection.

## For Beginners

Information Gain can be unfairly high for features with many
unique values (like IDs). Gain Ratio fixes this by dividing by how spread out the
feature itself is. A feature with 100 values needs more "justification" to score high
than one with just 2 values.

## How It Works

Gain Ratio normalizes Information Gain by the intrinsic value (entropy) of the feature
itself. This prevents bias toward features with many values and provides a more balanced
measure of feature relevance.

