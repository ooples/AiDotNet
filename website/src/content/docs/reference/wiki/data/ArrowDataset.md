---
title: "ArrowDataset<T>"
description: "Provides read/write access to datasets in a custom binary columnar format inspired by Apache Arrow IPC."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

Provides read/write access to datasets in a custom binary columnar format
inspired by Apache Arrow IPC.

## How It Works

**Important:** This is NOT a native Apache Arrow implementation and is NOT compatible with
Arrow IPC files created by PyArrow, the Arrow C++ library, or other Arrow tools. It is a custom
columnar format designed for efficient column-wise access in AiDotNet ML pipelines.
For native Arrow interop, use the Apache.Arrow NuGet package.

The on-disk format:

- Header: [magic: "ARRW" 4 bytes] [version: 4 bytes] [numRows: 4 bytes] [numColumns: 4 bytes]
- Column table: for each column: [nameLen: 4][name: bytes][elementType: 4][elementsPerRow: 4][dataOffset: 8][dataLength: 8]
- Data section: raw values for each column (doubles, 8 bytes each)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ArrowDataset(ArrowDatasetOptions)` | Opens an Arrow dataset file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnNames` | Gets the column names. |
| `NumRows` | Gets the total number of rows in the dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `ReadBatch(Int32,Int32)` | Reads feature and label columns as tensors. |
| `ReadColumn(String)` | Reads an entire column as a flat array. |
| `ReadColumnSlice(String,Int32,Int32)` | Reads a batch of rows from the specified columns. |
| `WriteFile(String,IReadOnlyDictionary<String,ValueTuple<[],Int32>>,Int32)` | Writes a dataset to Arrow format. |

