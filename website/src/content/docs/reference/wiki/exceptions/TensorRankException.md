---
title: "TensorRankException"
description: "Exception thrown when a tensor's rank doesn't match the expected rank."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when a tensor's rank doesn't match the expected rank.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorRankException(Int32,Int32,String,String)` | Initializes a new instance of the `TensorRankException` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualRank` | The actual rank of the tensor. |
| `Component` | The component where the rank mismatch occurred. |
| `ExpectedRank` | The expected rank of the tensor. |
| `Operation` | The operation being performed when the mismatch was detected. |

