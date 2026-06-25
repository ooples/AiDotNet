---
title: "SaliencySelector<T>"
description: "Saliency-based feature selection for image features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Image`

Saliency-based feature selection for image features.

## For Beginners

Saliency measures how much a small change in each
feature would affect the prediction. Features with high saliency are "sensitive"
to changes, suggesting they're important for the model's decisions.

## How It Works

Selects features based on gradient-based saliency, identifying features that
contribute most to model predictions. Originally designed for neural network
interpretability but adapted here for general feature selection.

