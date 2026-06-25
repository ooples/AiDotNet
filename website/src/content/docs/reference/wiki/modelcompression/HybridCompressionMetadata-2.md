---
title: "HybridCompressionMetadata<T>"
description: "Metadata for hybrid compression combining clustering and Huffman encoding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for hybrid compression combining clustering and Huffman encoding.

## For Beginners

This metadata combines information from both compression stages:

- Clustering metadata (cluster centers and assignments)
- Huffman metadata (encoding tree and table)

During decompression, Huffman decoding is applied first, then clustering decompression.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridCompressionMetadata(WeightClusteringMetadata<>,HuffmanEncodingMetadata<>,Int32)` | Initializes a new instance of the HybridCompressionMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusteringMetadata` | Gets the metadata from the clustering stage. |
| `HuffmanMetadata` | Gets the metadata from the Huffman encoding stage. |
| `OriginalLength` | Gets the original length of the weights array. |
| `Type` | Gets the compression type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

