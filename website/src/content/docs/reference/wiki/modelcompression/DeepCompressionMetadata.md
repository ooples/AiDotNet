---
title: "DeepCompressionMetadata<T>"
description: "Metadata for Deep Compression containing information from all three stages."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for Deep Compression containing information from all three stages.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepCompressionMetadata(SparsePruningMetadata<>,WeightClusteringMetadata<>,HuffmanEncodingMetadata<>,Int32,DeepCompressionStats)` | Initializes a new instance of the DeepCompressionMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusteringMetadata` | Gets the metadata from Stage 2 (Weight Clustering/Quantization). |
| `CompressionStats` | Gets the compression statistics. |
| `HuffmanMetadata` | Gets the metadata from Stage 3 (Huffman Encoding). |
| `OriginalLength` | Gets the original length of the weights array. |
| `PruningMetadata` | Gets the metadata from Stage 1 (Pruning). |
| `Type` | Gets the compression type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

