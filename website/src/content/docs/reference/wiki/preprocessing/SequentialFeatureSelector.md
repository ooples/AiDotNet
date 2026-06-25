---
title: "SequentialFeatureSelector<T>"
description: "Sequential Feature Selector using greedy forward or backward selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Sequential Feature Selector using greedy forward or backward selection.

## For Beginners

Think of forward selection like building a team: you start
with no players and add the best candidate one at a time until you have enough.
Backward selection is the opposite: start with everyone and cut the worst performer
one at a time. The "best" is determined by how well a model performs with those features.

## How It Works

Sequential Feature Selection (SFS) is a wrapper method that evaluates feature subsets
using a model's cross-validation score. Forward selection starts empty and adds features;
backward selection starts full and removes features.

