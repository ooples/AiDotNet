---
title: "MemoryMappedStreamingDataLoader<T, TInput, TOutput>"
description: "A streaming data loader that uses memory-mapped files for efficient random access to large binary datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

A streaming data loader that uses memory-mapped files for efficient random access
to large binary datasets.

## For Beginners

Memory-mapped files let the operating system manage
which parts of a large file are in memory. When you access a sample, the OS automatically
loads that portion of the file into RAM. This is very efficient for random access
patterns like shuffled batch iteration on datasets too large to fit in memory.

Example:

## How It Works

MemoryMappedStreamingDataLoader uses `MemoryMappedFile`
for efficient random access to large datasets stored in binary format. The operating system
handles paging data in and out of physical memory as needed, enabling efficient access to
datasets larger than available RAM.

**File Format Requirements:**

- Binary file with fixed-size samples
- Each sample is `inputSizeBytes + outputSizeBytes` bytes
- Samples are stored contiguously with optional header

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryMappedStreamingDataLoader(String,Int32,Int32,Int32,Func<Byte[],>,Func<Byte[],>,Int32,Int64,Int32,Int32)` | Initializes a new instance of the MemoryMappedStreamingDataLoader class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeaderSizeBytes` | Gets the size of the file header in bytes. |
| `Name` |  |
| `SampleCount` |  |
| `SampleSizeBytes` | Gets the size of each sample in bytes (input + output). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Releases all resources used by the memory-mapped data loader. |
| `Dispose(Boolean)` | Releases the unmanaged resources and optionally releases the managed resources. |
| `Finalize` | Finalizer to ensure resources are released. |
| `GetViewAccessor` | Gets the view accessor for reading from the memory-mapped file. |
| `ReadSampleAsync(Int32,CancellationToken)` |  |
| `UnloadDataCore` |  |

