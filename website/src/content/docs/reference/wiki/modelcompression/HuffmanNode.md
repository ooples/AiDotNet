---
title: "HuffmanNode<T>"
description: "Represents a node in the Huffman tree."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.ModelCompression`

Represents a node in the Huffman tree.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HuffmanNode(,Int32,Boolean,Int32,HuffmanNode<>,HuffmanNode<>)` | Initializes a new instance of the HuffmanNode class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Frequency` | Gets the frequency of this value or subtree. |
| `Id` | Gets the unique identifier for stable sorting. |
| `IsLeaf` | Gets a value indicating whether this is a leaf node. |
| `Left` | Gets the left child node (null for leaf nodes). |
| `Right` | Gets the right child node (null for leaf nodes). |
| `Value` | Gets the value stored in this node (for leaf nodes, default for internal nodes). |

