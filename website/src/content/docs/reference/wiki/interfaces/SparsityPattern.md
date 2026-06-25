---
title: "SparsityPattern"
description: "Types of sparsity patterns."
section: "API Reference"
---

`Enums` · `AiDotNet.Interfaces`

Types of sparsity patterns.

## Fields

| Field | Summary |
|:-----|:--------|
| `BlockStructured` | Block structured - dense blocks pruned together. |
| `ChannelStructured` | Channel-wise structured - entire channels removed. |
| `ColumnStructured` | Column-wise structured - entire columns removed. |
| `FilterStructured` | Filter-wise structured - entire conv filters removed. |
| `RowStructured` | Row-wise structured - entire rows removed. |
| `Structured2to4` | 2:4 fine-grained structured (NVIDIA Ampere). |
| `StructuredNtoM` | N:M fine-grained structured (generalized). |
| `Unstructured` | Unstructured - individual weights pruned randomly. |

