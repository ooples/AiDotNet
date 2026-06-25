---
title: "BanzhafFeatureSelector<T>"
description: "Banzhaf Power Index-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.GameTheory`

Banzhaf Power Index-based Feature Selection.

## For Beginners

The Banzhaf index measures how often adding a
feature changes the outcome (makes a coalition successful). Unlike Shapley,
it treats all coalition sizes equally. Features with high Banzhaf values are
"swing voters" that often make the difference.

## How It Works

Uses the Banzhaf power index from game theory to measure feature importance
based on the probability of being a swing voter (decisive feature).

