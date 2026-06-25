---
title: "RelationModuleType"
description: "Types of relation module architectures for Relation Networks."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Types of relation module architectures for Relation Networks.

## For Beginners

This determines HOW the network compares two examples.
Instead of using a fixed formula (like Euclidean distance), Relation Networks learn
a neural network to measure "how related" two examples are. This enum controls the
architecture of that comparison network.

## How It Works

The relation module learns to compare feature embeddings and output a similarity score.
Different architectures provide different ways of computing this learned similarity.

## Fields

| Field | Summary |
|:-----|:--------|
| `Attention` | Uses attention mechanism to relate features. |
| `Concatenate` | Concatenates features and passes through MLP. |
| `Convolution` | Stacks features and applies 2D convolution. |
| `Transformer` | Uses transformer-style self-attention. |

