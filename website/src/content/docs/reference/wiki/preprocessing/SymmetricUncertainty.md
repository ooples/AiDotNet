---
title: "SymmetricUncertainty<T>"
description: "Symmetric Uncertainty for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Symmetric Uncertainty for feature selection.

## For Beginners

Regular Mutual Information can be hard to interpret
because its scale depends on the data. Symmetric Uncertainty normalizes it to
always be between 0 (no relationship) and 1 (perfect relationship). It treats
both variables equally, so it doesn't matter which is the feature and which is
the target.

## How It Works

Symmetric Uncertainty is a normalized version of Mutual Information that ranges
from 0 to 1. It is symmetric, meaning SU(X,Y) = SU(Y,X), and corrects for the
bias toward features with many values.

