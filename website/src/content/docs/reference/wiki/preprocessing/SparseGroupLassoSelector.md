---
title: "SparseGroupLassoSelector<T>"
description: "Sparse Group Lasso Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sparse`

Sparse Group Lasso Feature Selection.

## For Beginners

Sometimes features naturally come in groups
(like one-hot encoded categories or polynomial terms). This method can select
entire groups together while also selecting individual features within groups,
giving you the best of both worlds.

## How It Works

Combines Lasso (individual feature sparsity) with Group Lasso (group sparsity)
to select features that are both individually and group-wise important.

