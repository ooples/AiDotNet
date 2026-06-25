---
title: "AssociativeMemory<T>"
description: "Implementation of Associative Memory for nested learning using modern continuous Hopfield retrieval (Ramsauer et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NestedLearning`

Implementation of Associative Memory for nested learning using modern continuous
Hopfield retrieval (Ramsauer et al. 2021). Stores key-value pairs and retrieves
via softmax attention: new_state = softmax(β * keys^T @ query) @ values.

## Properties

| Property | Summary |
|:-----|:--------|
| `MemoryCount` | Gets the number of stored memories. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAssociationMatrix` | Computes the Hebbian association matrix from stored memories: W = Σ target_i * input_i^T. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CosineEpsilon` | Small epsilon to prevent division by zero in cosine similarity. |
| `DuplicateKeyThreshold` | Cosine similarity threshold for treating two keys as duplicates in Update. |

