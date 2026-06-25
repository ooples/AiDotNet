---
title: "MemoryMappedDataset<T>"
description: "Memory-mapped dataset access for efficient I/O on large binary datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

Memory-mapped dataset access for efficient I/O on large binary datasets.

## For Beginners

Use this for very large datasets that don't fit in RAM.
The operating system will transparently page data in and out as needed.

## How It Works

Memory-mapped files allow the operating system to map file data directly into virtual memory,
enabling efficient access to large datasets without loading them entirely into RAM.
The OS handles paging data in and out as needed.

The dataset file uses a simple binary format:

- First 4 bytes: number of samples (int32 little-endian)
- Next 4 bytes: elements per sample (int32 little-endian)
- Remaining bytes: raw double-precision (8-byte) values, converted to/from T at read/write time.

When reading, this class materializes samples into managed arrays/tensors (not zero-copy),
but the OS-level memory mapping avoids reading the entire file into RAM upfront.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryMappedDataset(String)` | Creates a new memory-mapped dataset from a file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ElementsPerSample` | Gets the number of elements per sample. |
| `NumSamples` | Gets the total number of samples in the dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `EnumerateSamples` | Creates an iterator that yields all samples in sequential order. |
| `ReadBatch(Int32[],Int32[])` | Reads a batch of samples at the specified indices. |
| `ReadSample(Int32)` | Reads a single sample at the specified index. |
| `ReadSampleAsTensor(Int32,Int32[])` | Reads a single sample as a Tensor. |
| `WriteDatasetFile(String,Tensor<>)` | Writes a dataset file in the expected format. |

## Fields

| Field | Summary |
|:-----|:--------|
| `ElementSize` | Size in bytes of each element stored on disk (double = 8 bytes). |

