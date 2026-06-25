---
title: "SymmetricalUncertainty<T>"
description: "Symmetrical Uncertainty for feature selection based on normalized mutual information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Symmetrical Uncertainty for feature selection based on normalized mutual information.

## For Beginners

Mutual information tells you how much one variable reveals
about another, but the raw value depends on the variables' complexity. Symmetrical
Uncertainty fixes this by scaling the value between 0 and 1, where 0 means no
relationship and 1 means perfect predictability in both directions.

## How It Works

Symmetrical Uncertainty normalizes mutual information to the range [0, 1] by considering
the entropy of both variables. It measures how much knowing one variable reduces
uncertainty about the other, symmetrically.

