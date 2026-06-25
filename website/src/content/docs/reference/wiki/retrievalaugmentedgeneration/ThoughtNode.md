---
title: "ThoughtNode<T>"
description: "Represents a node in the Tree-of-Thoughts reasoning tree."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.RetrievalAugmentedGeneration.Models`

Represents a node in the Tree-of-Thoughts reasoning tree.

## Properties

| Property | Summary |
|:-----|:--------|
| `Children` | Child nodes branching from this thought. |
| `Depth` | Depth of this node in the tree (0 = root). |
| `EvaluationScore` | Evaluation score for this thought (0-1, higher is better). |
| `Parent` | Parent node in the tree (null for root). |
| `RetrievedDocuments` | Documents retrieved for this thought. |
| `Thought` | The reasoning thought or statement at this node. |

