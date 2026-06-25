---
title: "GradCAMSelector<T>"
description: "GradCAM++ based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Neural`

GradCAM++ based Feature Selection.

## For Beginners

GradCAM++ extends GradCAM by using weighted
combinations of positive partial derivatives. It identifies which features
the model focuses on when making predictions. Features with higher activation
weights are considered more important for the model's decisions.

## How It Works

Selects features using gradient-weighted class activation mapping,
measuring feature importance through gradient flow analysis.

