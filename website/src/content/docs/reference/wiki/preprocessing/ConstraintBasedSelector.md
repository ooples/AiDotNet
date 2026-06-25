---
title: "ConstraintBasedSelector<T>"
description: "Constraint-based Feature Selection with domain constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Constraint`

Constraint-based Feature Selection with domain constraints.

## For Beginners

Sometimes domain experts know certain features
must be included (like patient age in medical data) or must be excluded
(like sensitive personal info). This method respects those rules while
still finding the best possible feature subset.

## How It Works

Selects features while respecting domain-specific constraints such as
mandatory features that must be included, forbidden features that must
be excluded, and group constraints where features must be selected together.

