---
title: "IModelCompression<T, TMetadata>"
description: "Defines a type-safe interface for model compression used to reduce model size while preserving accuracy."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a type-safe interface for model compression used to reduce model size while preserving accuracy.

## For Beginners

Model compression makes AI models smaller and faster without significantly
hurting their performance.

Think of it like compressing a video file - you want to reduce the file size so it's easier to
store and share, but you still want the video to look good. Similarly, model compression reduces
the memory needed to store an AI model and can make it run faster, which is especially important
for deploying models on mobile devices or in the cloud where storage and processing costs matter.

Different compression strategies work in different ways:

- Some group similar values together (clustering)
- Some remove less important parts (pruning)
- Some use clever encoding schemes to store data more efficiently (quantization, Huffman coding)

This interface defines the standard methods that all compression implementations must provide.
The TMetadata type parameter ensures that each compression algorithm uses its own specific
metadata type, preventing errors from using the wrong metadata with the wrong algorithm.

## How It Works

This interface provides type-safe model compression by using strongly-typed metadata instead of
the object type. Each compression algorithm defines its own metadata class that implements
`ICompressionMetadata`, ensuring compile-time type safety.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCompressionRatio(Int64,Int64)` | Calculates the compression ratio achieved. |
| `Compress(Vector<>)` | Compresses the given model weights. |
| `Decompress(Vector<>,)` | Decompresses the compressed weights back to their original form. |
| `GetCompressedSize(Vector<>,)` | Gets the size in bytes of the compressed representation. |

