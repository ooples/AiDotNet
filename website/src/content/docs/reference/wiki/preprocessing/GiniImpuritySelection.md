---
title: "GiniImpuritySelection<T>"
description: "Gini Impurity-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Gini Impurity-based Feature Selection.

## For Beginners

Gini impurity measures how "mixed" a group is.
If splitting data by a feature creates groups where samples are mostly
the same class (pure groups), that feature is valuable. This method finds
features that best separate different classes from each other.

## How It Works

Measures how much each feature reduces impurity (Gini impurity) when used
to split the data. Features that create purer groups after splitting are
considered more important.

