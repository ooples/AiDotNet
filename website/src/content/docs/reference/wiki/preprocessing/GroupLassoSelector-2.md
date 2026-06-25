---
title: "GroupLassoSelector<T>"
description: "Group Lasso based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sparsity`

Group Lasso based Feature Selection.

## For Beginners

Group Lasso extends Lasso by treating features
in groups. Instead of selecting individual features, it selects entire groups
at once. This is useful when features naturally belong together, like one-hot
encoded categories or polynomial terms of the same variable.

## How It Works

Selects features using Group Lasso regularization, which selects or
removes entire groups of features together.

