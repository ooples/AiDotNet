---
title: "BayesianModelAveraging<T>"
description: "Bayesian Model Averaging Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bayesian`

Bayesian Model Averaging Feature Selection.

## For Beginners

Instead of picking one best model, BMA considers
all possible models and weights them by how likely each is. Features that
appear in many high-probability models are selected.

## How It Works

Selects features based on their posterior inclusion probabilities across
all possible models, weighted by model posterior probabilities.

