---
title: "CompressionHelper"
description: "Provides transparent compression and decompression utilities for model serialization."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides transparent compression and decompression utilities for model serialization.

## For Beginners

This helper automatically compresses your model when you save it
and decompresses it when you load it. You don't need to do anything special - just configure
compression when building your model, and the rest happens automatically.

Benefits:

- Smaller model files (50-90% size reduction)
- Faster model loading over networks
- Lower storage costs
- Seamless integration with existing serialize/deserialize methods

## How It Works

CompressionHelper handles the application of compression during model serialization and
decompression during deserialization. It works transparently with the facade pattern,
so users never directly interact with compression algorithms.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCompression(Byte[],CompressionType,CompressionConfig)` | Applies the specified compression algorithm to the data. |
| `ApplyDecompression(Byte[],CompressionType,Int32)` | Applies the corresponding decompression algorithm. |
| `ChooseOptimalCompression(Byte[],CompressionConfig)` | Chooses the optimal compression type based on data characteristics. |
| `Compress(Byte[],CompressionConfig)` | Compresses the serialized model data based on the compression configuration. |
| `CompressWithBrotli(Byte[],CompressionLevel)` | Compresses data using the Brotli algorithm. |
| `CompressWithDeflate(Byte[],CompressionLevel)` | Compresses data using the Deflate algorithm. |
| `CompressWithGZip(Byte[],CompressionLevel)` | Compresses data using the GZip algorithm. |
| `Decompress(Byte[])` | Decompresses the model data. |
| `DecompressIfNeeded(Byte[])` | Decompresses model data if it was compressed, otherwise returns unchanged data. |
| `DecompressWithBrotli(Byte[])` | Decompresses data using the Brotli algorithm. |
| `DecompressWithDeflate(Byte[])` | Decompresses data using the Deflate algorithm. |
| `DecompressWithGZip(Byte[])` | Decompresses data using the GZip algorithm. |
| `GetCompressionStats(Byte[],Byte[])` | Gets compression statistics for the last compression operation. |
| `IsCompressedData(Byte[])` | Checks if the data was compressed by AiDotNet. |

## Fields

| Field | Summary |
|:-----|:--------|
| `FormatVersion` | Current compression format version for forward compatibility. |
| `MagicBytes` | Magic bytes to identify compressed model data. |

