---
title: "SparseCompressionResult<T>"
description: "Result of sparse compression operation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Result of sparse compression operation.

## Properties

| Property | Summary |
|:-----|:--------|
| `BlockIndices` | Block indices (for block-sparse format). |
| `BlockSize` | Block size for block-sparse format. |
| `ColumnIndices` | Column indices (for COO, CSC formats). |
| `ColumnPointers` | Column pointers (for CSC format). |
| `Format` | The sparse format used. |
| `NonZeroCount` | Number of non-zero elements. |
| `OriginalShape` | Original dense shape. |
| `RowIndices` | Row indices (for COO, CSR formats). |
| `RowPointers` | Row pointers (for CSR format). |
| `Sparsity` | Sparsity ratio (fraction of zeros). |
| `SparsityM` | M value for N:M sparsity patterns. |
| `SparsityMask` | Mask for 2:4 or N:M structured sparsity. |
| `SparsityN` | N value for N:M sparsity patterns. |
| `Values` | Non-zero values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetCompressedSizeBytes(Int32)` | Gets the compressed size in bytes. |

