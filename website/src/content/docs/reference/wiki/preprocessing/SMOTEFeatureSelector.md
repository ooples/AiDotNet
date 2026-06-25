---
title: "SMOTEFeatureSelector<T>"
description: "SMOTE-aware Feature Selection for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Imbalanced`

SMOTE-aware Feature Selection for imbalanced datasets.

## For Beginners

When you have few examples of one class (like rare diseases),
regular feature selection might ignore patterns specific to that class. This method
creates synthetic examples of the minority class first, then selects features that
distinguish both classes well.

## How It Works

This feature selector is designed for imbalanced datasets where one class is
significantly underrepresented. It evaluates features after synthetic minority
oversampling to ensure selected features work well on balanced data.

