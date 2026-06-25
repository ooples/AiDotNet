---
title: "HybridHuffmanClusteringCompression<T>"
description: "HybridHuffmanClusteringCompression<T> — Models & Types in AiDotNet.ModelCompression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridHuffmanClusteringCompression(Int32,Int32,Double,Int32,Nullable<Int32>)` | Initializes a new instance of the HybridHuffmanClusteringCompression class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(Vector<>)` | Compresses weights using clustering followed by Huffman encoding. |
| `Decompress(Vector<>,ICompressionMetadata<>)` | Decompresses weights by reversing Huffman encoding then clustering. |
| `GetCompressedSize(Vector<>,ICompressionMetadata<>)` | Gets the total compressed size from both compression stages. |

