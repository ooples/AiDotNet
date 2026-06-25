---
title: "ConditionalInferenceTreeNode<T>"
description: "Represents a node in a conditional inference tree, which is a type of decision tree that uses statistical tests to make decisions at each node."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.LinearAlgebra`

Represents a node in a conditional inference tree, which is a type of decision tree
that uses statistical tests to make decisions at each node.

## For Beginners

Think of this as a special type of decision tree that makes decisions
based on statistical evidence rather than just information gain. The p-value stored in each
node represents how confident we are that the split at this node is meaningful and not just
due to random chance. Lower p-values (closer to zero) indicate stronger evidence for the split.

## How It Works

A conditional inference tree is a statistical approach to decision tree learning that
uses significance tests to select variables at each split.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConditionalInferenceTreeNode` | Initializes a new instance of the `ConditionalInferenceTreeNode` class with a default p-value of zero. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PValue` | Gets or sets the p-value associated with the statistical test at this node. |

