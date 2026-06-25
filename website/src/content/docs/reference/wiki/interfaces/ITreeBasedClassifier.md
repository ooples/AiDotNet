---
title: "ITreeBasedClassifier<T>"
description: "Interface for tree-based classification algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for tree-based classification algorithms.

## For Beginners

Decision trees work like a flowchart - they ask a series of yes/no questions about features
to reach a decision. For example, to classify if an animal is a cat:
"Has fur?" (yes) -> "Has whiskers?" (yes) -> "Meows?" (yes) -> "It's a cat!"

Key properties:

- MaxDepth: How deep the tree can go (more depth = more complex decisions)
- Feature importance: Which features were most useful for classification

## How It Works

Tree-based classifiers make decisions by learning a series of hierarchical rules from data.
They are highly interpretable and can capture non-linear relationships between features.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureImportances` | Gets the feature importance scores computed during training. |
| `LeafCount` | Gets the number of leaf nodes in the tree. |
| `MaxDepth` | Gets the maximum depth of the tree. |
| `NodeCount` | Gets the number of internal (decision) nodes in the tree. |

