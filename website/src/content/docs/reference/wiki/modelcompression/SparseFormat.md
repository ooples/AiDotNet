---
title: "SparseFormat"
description: "Sparse storage formats."
section: "API Reference"
---

`Enums` · `AiDotNet.ModelCompression`

Sparse storage formats.

## Fields

| Field | Summary |
|:-----|:--------|
| `BSR` | Block Sparse Row (BSR) - like CSR but with dense blocks. |
| `COO` | Coordinate format (COO) - stores (row, col, value) triplets. |
| `CSC` | Compressed Sparse Column (CSC) - column pointers + row indices + values. |
| `CSR` | Compressed Sparse Row (CSR) - row pointers + column indices + values. |
| `DIA` | Diagonal format - for diagonal or banded matrices. |
| `ELL` | ELLPACK format - fixed number of non-zeros per row. |
| `Structured2to4` | 2:4 Structured Sparsity - 2 zeros per 4 elements (NVIDIA Ampere compatible). |
| `StructuredNtoM` | N:M Fine-grained structured sparsity (generalized). |

