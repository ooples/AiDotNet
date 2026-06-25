---
title: "CorrelationMIHybrid<T>"
description: "Correlation-Mutual Information Hybrid feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Correlation-Mutual Information Hybrid feature selection.

## For Beginners

Correlation finds straight-line relationships
while mutual information finds any type of pattern. A feature might be
strongly related to the target in a curved way that correlation misses.
This hybrid catches both types of useful features.

## How It Works

Combines linear correlation (Pearson) with mutual information (non-linear)
to capture both types of feature-target relationships.

