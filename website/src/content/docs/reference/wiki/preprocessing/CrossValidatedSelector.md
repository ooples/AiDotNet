---
title: "CrossValidatedSelector<T>"
description: "Cross-Validated Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.CrossValidation`

Cross-Validated Feature Selection.

## For Beginners

Cross-validation tests how well features predict
targets on data not used for training. Features that consistently help
predictions across multiple data splits are selected.

## How It Works

Selects features based on their predictive performance evaluated using
k-fold cross-validation with a simple linear model.

