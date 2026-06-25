---
title: "TreeBasedImportance<T>"
description: "Tree-based feature importance using random forest mean decrease impurity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Tree-based feature importance using random forest mean decrease impurity.

## For Beginners

Random forests build many decision trees, and each tree
splits the data at various points. When a feature is used for a split, it reduces
the "messiness" (impurity) of the data. Features that consistently help clean up
the mess get high importance scores.

## How It Works

Tree-based importance measures the average reduction in impurity (Gini or entropy)
achieved by splits on each feature across all trees in a random forest. Features
that are used for impactful splits get higher importance scores.

