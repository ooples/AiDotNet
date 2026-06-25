---
title: "RandomForestImportance<T>"
description: "Random Forest feature importance for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Random Forest feature importance for feature selection.

## For Beginners

Random forests are ensembles of decision trees.
When building each tree, features that better split the data are used more
often and at higher positions. Feature importance is measured by how much
each feature helps reduce errors across all trees.

## How It Works

Builds a simplified random forest and measures feature importance based on
how much each feature contributes to reducing prediction error (Gini importance
or permutation importance).

