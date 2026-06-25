---
title: "ExhaustiveFeatureSelector<T>"
description: "Exhaustive Feature Selector that evaluates all possible feature subsets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Exhaustive Feature Selector that evaluates all possible feature subsets.

## For Beginners

This is the brute-force approach: try every possible
combination of features and pick the best one. It's guaranteed to find the optimal
subset, but it's only practical for small numbers of features (typically under 20)
because the number of combinations grows exponentially.

## How It Works

The Exhaustive Feature Selector evaluates all possible combinations of features up to
a maximum subset size. This guarantees finding the optimal subset but has exponential
complexity O(2^p) where p is the number of features.

