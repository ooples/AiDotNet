---
title: "HuffmanEncodingMetadata<T>"
description: "Metadata for Huffman encoding compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for Huffman encoding compression.

## For Beginners

This metadata stores the information needed to decompress Huffman-encoded weights:

- The Huffman tree (for decoding variable-length codes back to values)
- The encoding table (mapping values to their codes, used during compression)
- The original length and bit length (for proper reconstruction)

Huffman encoding is lossless - you get exactly the original values back when decompressing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HuffmanEncodingMetadata(HuffmanNode<>,NumericDictionary<,String>,Int32,Int32)` | Initializes a new instance of the HuffmanEncodingMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitLength` | Gets the length of the encoded bit stream. |
| `EncodingTable` | Gets the encoding table mapping values to codes. |
| `HuffmanTree` | Gets the Huffman tree used for encoding. |
| `OriginalLength` | Gets the original length of the weights vector. |
| `Type` | Gets the compression type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

