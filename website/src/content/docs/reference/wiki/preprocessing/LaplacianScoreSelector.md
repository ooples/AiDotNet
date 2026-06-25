---
title: "LaplacianScoreSelector<T>"
description: "Laplacian Score Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Manifold`

Laplacian Score Feature Selection.

## For Beginners

The Laplacian score measures how much a feature
changes between nearby points. Features that are smooth (don't change much
between neighbors) have low Laplacian scores and are considered more
important for preserving local structure.

## How It Works

Uses the Laplacian Score to select features that best preserve the local
structure of the data. Features with lower Laplacian scores are considered
more important.

