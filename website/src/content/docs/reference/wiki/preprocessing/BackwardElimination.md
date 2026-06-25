---
title: "BackwardElimination<T>"
description: "Backward Elimination (Sequential Backward Selection) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Backward Elimination (Sequential Backward Selection) for feature selection.

## For Beginners

Backward Elimination is like downsizing a team. You
start with everyone and repeatedly remove the person whose absence hurts performance
least. Unlike Forward Selection, this can find features that are only valuable
together, since they start included.

## How It Works

Starts with all features and greedily removes one feature at a time whose removal
causes the least degradation in performance until the desired number is reached.

