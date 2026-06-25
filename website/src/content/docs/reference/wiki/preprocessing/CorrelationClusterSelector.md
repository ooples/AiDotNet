---
title: "CorrelationClusterSelector<T>"
description: "Correlation Cluster based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Redundancy`

Correlation Cluster based Feature Selection.

## For Beginners

Features that are highly correlated form natural
groups. Instead of keeping all similar features, this selector picks one
representative from each group, reducing redundancy while keeping diversity.

## How It Works

Selects features by clustering correlated features together and selecting
the best representative from each cluster.

