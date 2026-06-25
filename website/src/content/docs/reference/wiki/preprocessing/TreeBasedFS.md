---
title: "TreeBasedFS<T>"
description: "Tree-Based Feature Selection using decision tree splits."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Tree-Based Feature Selection using decision tree splits.

## For Beginners

Decision trees ask questions like "is feature X
greater than value V?" to split data. Features that lead to purer groups
(where samples in each group are mostly the same class) are more important.
This method finds which features make the best splitting questions.

## How It Works

Builds a simple decision tree and measures feature importance by how much
each feature improves predictions when used for splitting. Features that
create better splits are considered more important.

