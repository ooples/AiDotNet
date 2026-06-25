---
title: "ICompressionMetadata<T>"
description: "Defines the contract for compression metadata that stores information needed to decompress model weights."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for compression metadata that stores information needed to decompress model weights.

## For Beginners

When you compress something, you need to remember how you compressed it
so you can undo it later. This metadata is like a "recipe" for decompression.

For example, if you compress weights using clustering:

- The compressed data contains which cluster each weight belongs to (just a number like 0, 1, 2...)
- The metadata contains the actual cluster center values (like 0.5, 1.2, 3.7...)
- To decompress, you look up each cluster number and replace it with the actual value

Without this metadata, you couldn't restore the original weights from the compressed data.

## How It Works

Compression metadata contains the essential information required to reverse the compression process.
Different compression algorithms produce different types of metadata - for example, weight clustering
stores cluster centers, while Huffman encoding stores the encoding tree.

## Properties

| Property | Summary |
|:-----|:--------|
| `OriginalLength` | Gets the original length of the uncompressed weight vector. |
| `Type` | Gets the type of compression algorithm that produced this metadata. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

