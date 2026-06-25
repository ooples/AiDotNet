---
title: "LLEBasedSelector<T>"
description: "Locally Linear Embedding (LLE) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Manifold`

Locally Linear Embedding (LLE) based Feature Selection.

## For Beginners

LLE assumes data lies on a curved surface
(manifold) in high-dimensional space. It finds how to reconstruct each
point from its neighbors. Features that are most important for these
local reconstructions are selected.

## How It Works

Uses Locally Linear Embedding to identify features that best preserve
the local neighborhood structure of the data manifold.

