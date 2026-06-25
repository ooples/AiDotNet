---
title: "MatrixCompressionMetadata<T>"
description: "Metadata for matrix compression operations that wraps the underlying vector compression metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for matrix compression operations that wraps the underlying vector compression metadata.

## For Beginners

When compressing a 2D matrix (like weights in a fully connected layer),
we need to remember:

1. The original shape - how many rows and columns the matrix had
2. How the flattened data was compressed (the inner compression details)

Think of it like folding a shirt to pack in a suitcase:

- You flatten the shirt (2D to 1D)
- You compress it in a vacuum bag (apply compression algorithm)
- You need to remember the original shirt size to unfold it properly later

This metadata class keeps track of all that information so we can perfectly restore
the original matrix shape after decompression.

## How It Works

MatrixCompressionMetadata stores the information needed to decompress a 2D weight matrix that was
compressed by first flattening it to a vector. It preserves the original matrix dimensions and
delegates the actual compression metadata to an inner ICompressionMetadata instance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatrixCompressionMetadata(Int32,Int32,ICompressionMetadata<>)` | Initializes a new instance of the MatrixCompressionMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InnerMetadata` | Gets the inner compression metadata from the underlying vector compression algorithm. |
| `OriginalColumns` | Gets the number of columns in the original matrix. |
| `OriginalLength` | Gets the original total number of elements in the flattened matrix. |
| `OriginalRows` | Gets the number of rows in the original matrix. |
| `Type` | Gets the compression type from the underlying compression algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the total size in bytes of this metadata structure, including the inner metadata. |

