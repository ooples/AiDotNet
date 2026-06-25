---
title: "DropColumnImportance<T>"
description: "Drop Column Importance feature selection by measuring score decrease when dropping features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Drop Column Importance feature selection by measuring score decrease when dropping features.

## For Beginners

Instead of shuffling a feature (like permutation importance),
you completely remove it and retrain the model from scratch. If the model performs much
worse without a feature, that feature is important for learning. This is more thorough
but also more computationally expensive since you retrain for each feature.

## How It Works

Drop Column Importance measures the decrease in model performance when each feature is
completely removed from the dataset and the model is retrained. Unlike permutation
importance, this captures the feature's contribution to model learning, not just prediction.

