---
title: "CFS<T>"
description: "Correlation-based Feature Selection (CFS) for selecting feature subsets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Correlation-based Feature Selection (CFS) for selecting feature subsets.

## For Beginners

CFS looks for features that are good at predicting
the target but aren't redundant with each other. If two features give the same
information, CFS keeps only one. This prevents selecting features that are
copies of each other while maximizing predictive power.

## How It Works

CFS evaluates feature subsets based on a heuristic: good feature sets contain
features highly correlated with the class but uncorrelated with each other.
Uses greedy forward selection with the CFS merit function.

