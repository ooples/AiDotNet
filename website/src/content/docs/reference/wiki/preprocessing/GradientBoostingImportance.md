---
title: "GradientBoostingImportance<T>"
description: "Gradient Boosting-based feature importance using iterative boosting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Gradient Boosting-based feature importance using iterative boosting.

## For Beginners

Gradient Boosting builds trees sequentially, where
each tree tries to fix the errors of previous trees. The features that help the
most in reducing these errors across all trees are considered important.

## How It Works

Gradient Boosting importance measures how often each feature is used for splits
and how much those splits contribute to model improvement across all boosting
rounds. Features that are frequently used for impactful splits score higher.

