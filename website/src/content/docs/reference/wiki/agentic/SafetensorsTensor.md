---
title: "SafetensorsTensor"
description: "Metadata for one tensor inside a safetensors file: its name, dtype, shape, and byte range within the data section."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

Metadata for one tensor inside a safetensors file: its name, dtype, shape, and byte range within the data
section.

## For Beginners

A safetensors file stores many named weight arrays back to back. This is the
"table of contents" entry for one of them — what it's called, its numeric type, its dimensions, and where
its bytes live in the file.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SafetensorsTensor(String,String,IReadOnlyList<Int64>,Int64,Int64)` | Initializes a new tensor descriptor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BeginByte` | Gets the start offset within the data section (inclusive). |
| `ByteLength` | Gets the number of bytes occupied by this tensor. |
| `DataType` | Gets the safetensors dtype string. |
| `ElementCount` | Gets the total element count (product of `Shape`; 1 for a scalar). |
| `EndByte` | Gets the end offset within the data section (exclusive). |
| `Name` | Gets the tensor name. |
| `Shape` | Gets the tensor shape. |

