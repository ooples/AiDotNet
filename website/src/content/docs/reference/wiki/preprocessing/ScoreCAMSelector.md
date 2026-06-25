---
title: "ScoreCAMSelector<T>"
description: "Score-CAM based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Neural`

Score-CAM based Feature Selection.

## For Beginners

Unlike GradCAM which uses gradients, Score-CAM
measures feature importance by masking each feature and observing how the
prediction score changes. Features that cause larger score drops when
masked are considered more important. This is more stable than gradient methods.

## How It Works

Selects features using Score-CAM, a gradient-free class activation mapping
method that uses forward passing scores instead of gradients.

