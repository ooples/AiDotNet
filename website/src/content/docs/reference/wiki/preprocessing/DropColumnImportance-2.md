---
title: "DropColumnImportance<T>"
description: "Drop-Column Importance for model-agnostic feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic`

Drop-Column Importance for model-agnostic feature selection.

## For Beginners

This method is like permutation importance but more
definitive. Instead of scrambling a feature, it completely removes it and
retrains the model. If the model gets much worse without a feature, that
feature was truly important.

## How It Works

Drop-Column Importance measures feature importance by removing each feature
entirely and measuring how much model performance degrades. This is more
thorough than permutation but requires retraining the model for each feature.

