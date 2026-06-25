---
title: "BoostingFeatureSelector<T>"
description: "Boosting-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Boosting-based Feature Selection.

## For Beginners

Like gradient boosting, this method focuses more
attention on the samples that are hardest to predict. Features that help
predict these difficult samples get higher importance scores, leading to
more discriminative feature selection.

## How It Works

Uses a boosting-inspired approach where samples are reweighted based on
prediction errors, focusing feature selection on harder-to-predict samples.

