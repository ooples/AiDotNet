---
title: "DecisionTreeClassifierOptions<T>"
description: "Configuration options for decision tree classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for decision tree classifiers.

## For Beginners

Decision trees are like a game of 20 questions.

At each step, the tree asks a question about a feature:
"Is age > 30?" -> Yes: "Is income > 50000?" -> No: "Deny loan"
-> No: "Is student?" -> Yes: "Approve loan"

Key settings:

- MaxDepth: Limits how many questions deep the tree can go
- MinSamplesSplit: Minimum samples needed to continue splitting
- MaxFeatures: How many features to consider at each split

## How It Works

Decision trees are supervised learning algorithms that learn a hierarchy of
if/else rules from training data. They are easy to interpret and can handle
both numerical and categorical features.

## Properties

| Property | Summary |
|:-----|:--------|
| `Criterion` | Gets or sets the criterion used to measure the quality of a split. |
| `MaxDepth` | Gets or sets the maximum depth of the tree. |
| `MaxFeatures` | Gets or sets the number of features to consider when looking for the best split. |
| `MinImpurityDecrease` | Gets or sets the minimum impurity decrease required for a split. |
| `MinSamplesLeaf` | Gets or sets the minimum number of samples required at a leaf node. |
| `MinSamplesSplit` | Gets or sets the minimum number of samples required to split an internal node. |

