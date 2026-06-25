---
title: "SparseGroupLasso<T>"
description: "Sparse Group Lasso for feature selection with grouped structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sparse`

Sparse Group Lasso for feature selection with grouped structure.

## For Beginners

Imagine features organized into categories (e.g., color
features, shape features). Sometimes you want to select entire categories, other times
specific features within categories. Sparse Group Lasso does both: it can eliminate
entire groups or pick individual features within groups.

## How It Works

Sparse Group Lasso combines L1 (Lasso) and L2 (Group Lasso) penalties to select
both entire groups of features and individual features within groups. This is useful
when features have a natural grouping structure.

