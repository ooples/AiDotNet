---
title: "InformationGainRatioSelector<T>"
description: "Information Gain Ratio based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Gain`

Information Gain Ratio based Feature Selection.

## For Beginners

Information gain measures how much knowing a
feature reduces uncertainty about the target. But features with many values
(like IDs) can have artificially high gain. Gain ratio fixes this by dividing
by the feature's own entropy, giving a fair comparison across features.

## How It Works

Selects features based on information gain ratio, which normalizes
information gain by the feature's intrinsic information to avoid
bias toward features with many distinct values.

