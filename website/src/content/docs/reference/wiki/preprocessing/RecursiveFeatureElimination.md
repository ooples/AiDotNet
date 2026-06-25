---
title: "RecursiveFeatureElimination<T>"
description: "Recursive Feature Elimination (RFE) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Recursive Feature Elimination (RFE) for feature selection.

## For Beginners

RFE works like an elimination tournament. It trains
a model, finds the weakest feature, removes it, and repeats. This continues until
only the strongest features remain. It's more thorough than one-shot methods
because feature importance changes as features are removed.

## How It Works

RFE recursively removes the least important features based on model weights or
feature importances. At each step, a model is trained and the feature with the
lowest importance is eliminated until the desired number of features remains.

