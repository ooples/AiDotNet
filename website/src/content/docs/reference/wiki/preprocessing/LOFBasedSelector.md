---
title: "LOFBasedSelector<T>"
description: "Local Outlier Factor (LOF) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Density`

Local Outlier Factor (LOF) based Feature Selection.

## For Beginners

LOF measures how "outlying" each point is compared
to its neighbors. This selector chooses features that help distinguish normal
points from outliers, preserving the local density patterns in your data.

## How It Works

Uses the Local Outlier Factor concept to select features that best preserve
local density structure, helping maintain meaningful neighborhoods.

