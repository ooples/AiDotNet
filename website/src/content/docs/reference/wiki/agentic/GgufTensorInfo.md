---
title: "GgufTensorInfo"
description: "A tensor directory entry in a GGUF file: its name, dimensions, ggml data type, and byte offset within the tensor-data section."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

A tensor directory entry in a GGUF file: its name, dimensions, ggml data type, and byte offset within the
tensor-data section.

## For Beginners

GGUF (the llama.cpp weight format) lists every weight array with its shape,
numeric type, and where its bytes start. This is one such listing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GgufTensorInfo(String,IReadOnlyList<Int64>,UInt32,UInt64)` | Initializes a new tensor info. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Dimensions` | Gets the tensor dimensions. |
| `ElementCount` | Gets the total element count (product of `Dimensions`; validated at construction). |
| `GgmlType` | Gets the ggml type code (0 = F32, 1 = F16, higher = quantized formats). |
| `Name` | Gets the tensor name. |
| `Offset` | Gets the byte offset of the tensor within the data section. |

## Fields

| Field | Summary |
|:-----|:--------|
| `QuantBlockSize` | The number of values per quantization block (ggml QK). |
| `SuperBlockSize` | The number of values per k-quant super-block (ggml QK_K). |
| `TypeF16` | The ggml type code for 16-bit float tensors. |
| `TypeF32` | The ggml type code for 32-bit float tensors. |
| `TypeQ4_0` | The ggml type code for Q4_0 quantization (32-value blocks, 4-bit, single scale). |
| `TypeQ4_1` | The ggml type code for Q4_1 quantization (32-value blocks, 4-bit, scale + min). |
| `TypeQ4_K` | The ggml type code for Q4_K quantization (256-value super-blocks, 4-bit, 8 sub-scales/mins). |
| `TypeQ6_K` | The ggml type code for Q6_K quantization (256-value super-blocks, 6-bit, 16 sub-scales). |
| `TypeQ8_0` | The ggml type code for Q8_0 quantization (32-value blocks, 8-bit, single scale). |

