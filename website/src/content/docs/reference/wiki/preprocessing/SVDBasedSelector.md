---
title: "SVDBasedSelector<T>"
description: "SVD-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Projection`

SVD-based Feature Selection.

## For Beginners

SVD breaks down your data into components ordered
by importance. Features that have high values in the most important components
are the features that capture the most information about the data's structure.

## How It Works

Uses Singular Value Decomposition to identify features that contribute
most to the dominant singular vectors.

