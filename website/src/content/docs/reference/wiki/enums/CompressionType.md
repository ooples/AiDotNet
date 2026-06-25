---
title: "CompressionType"
description: "Defines the types of model compression strategies available in the AiDotNet library."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the types of model compression strategies available in the AiDotNet library.

## For Beginners

Model compression reduces the size of AI models while trying to maintain their accuracy.
Think of it like compressing a photo - you want a smaller file size but still a recognizable image.
Different compression techniques work better for different scenarios and model types.

## Fields

| Field | Summary |
|:-----|:--------|
| `DeepCompression` | Deep Compression combines pruning, quantization, and Huffman coding (Han et al. |
| `HuffmanEncoding` | Huffman encoding uses variable-length codes where frequent values get shorter codes. |
| `HybridClusteringPruning` | Combines weight clustering with pruning (removing unimportant weights). |
| `HybridClusteringQuantization` | Combines weight clustering with quantization for improved compression. |
| `HybridHuffmanClustering` | Combines Huffman encoding with weight clustering for maximum compression. |
| `LowRankFactorization` | Low-rank matrix factorization approximates weight matrices with lower-rank representations. |
| `None` | No compression applied to the model. |
| `ProductQuantization` | Product quantization divides weight vectors into sub-vectors and quantizes each separately. |
| `SparsePruning` | Sparse pruning removes small-magnitude weights, setting them to zero. |
| `WeightClustering` | Weight clustering groups similar weight values together and replaces them with cluster representatives. |

