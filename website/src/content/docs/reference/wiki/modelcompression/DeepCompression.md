---
title: "DeepCompression<T>"
description: "Implements the Deep Compression algorithm from Han et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Implements the Deep Compression algorithm from Han et al. (2015).

## For Beginners

Deep Compression is like a three-step recipe for making
neural networks much smaller:

**Step 1 - Pruning (Remove the unimportant)**
Think of it like cleaning out your closet. Many neural network weights are tiny
and don't really matter. We set these to zero and don't store them at all.
This alone can give ~9x compression!

**Step 2 - Quantization (Group similar values)**
After pruning, we group similar weight values together. Instead of storing the
exact value 0.4523, 0.4518, 0.4531, we store them all as "cluster #7 = 0.4524".
We only need to store which cluster each weight belongs to.
This gives another ~4x compression!

**Step 3 - Huffman Coding (Efficient storage)**
Some cluster numbers appear more often than others. We use shorter codes for
common values and longer codes for rare values (like Morse code).
This gives another ~1.5x compression!

Combined: 9 × 4 × 1.5 ≈ 35-50x compression!

Example usage:

## How It Works

Deep Compression is a three-stage compression pipeline that achieves 35-49x compression
on neural networks with minimal accuracy loss. The technique was introduced in:

Han, S., Mao, H., & Dally, W. J. (2015). "Deep Compression: Compressing Deep Neural
Networks with Pruning, Trained Quantization and Huffman Coding." arXiv:1510.00149.

The three stages are applied sequentially:

1. **Pruning**: Remove weights below a magnitude threshold
2. **Quantization**: Cluster remaining weights using k-means (weight sharing)
3. **Huffman Coding**: Apply entropy coding to the sparse, quantized representation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepCompression(Double,Double,Int32,Int32,Double,Int32,Nullable<Int32>,Boolean)` | Initializes a new instance of the DeepCompression class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCompressionStats(Vector<>,Vector<>,SparsePruningMetadata<>,WeightClusteringMetadata<>,HuffmanEncodingMetadata<>)` | Calculates compression statistics for the Deep Compression pipeline. |
| `Compress(Vector<>)` | Compresses weights using the three-stage Deep Compression pipeline. |
| `Decompress(Vector<>,ICompressionMetadata<>)` | Decompresses weights by reversing all three stages. |
| `ForConvolutionalLayers(Nullable<Int32>)` | Creates a DeepCompression instance optimized for convolutional layers. |
| `ForFullyConnectedLayers(Nullable<Int32>)` | Creates a DeepCompression instance optimized for fully-connected layers. |
| `GetCompressedSize(Vector<>,ICompressionMetadata<>)` | Gets the total compressed size from all three stages. |

