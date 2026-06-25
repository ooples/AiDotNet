---
title: "CompressedTree<T>"
description: "Represents a compressed decision-tree encoding of a parameter update."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Represents a compressed decision-tree encoding of a parameter update.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompressedTree(Dictionary<String,Double[]>,Dictionary<String,Double[]>,Dictionary<String,Int32[]>)` | Creates a new compressed tree. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LeafCounts` | Number of parameters per leaf per layer. |
| `LeafValues` | Leaf average values per layer. |
| `SplitPoints` | Split thresholds per layer. |

