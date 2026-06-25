---
title: "IModelCompressionStrategy<T>"
description: "Defines an interface for model compression strategies used to reduce model size while preserving accuracy."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for model compression strategies used to reduce model size while preserving accuracy.

## For Beginners

Model compression is the process of making AI models smaller and faster
without significantly hurting their performance.

Think of it like compressing a video file - you want to reduce the file size so it's easier to
store and share, but you still want the video to look good. Similarly, model compression reduces
the memory needed to store an AI model and can make it run faster, which is especially important
for deploying models on mobile devices or in the cloud where storage and processing costs matter.

Different compression strategies work in different ways:

- Some group similar values together (clustering)
- Some remove less important parts (pruning)
- Some use clever encoding schemes to store data more efficiently (quantization, Huffman coding)

This interface supports ALL types of neural network data:

- Vectors (1D): Bias terms, embeddings
- Matrices (2D): Fully connected layer weights
- Tensors (N-D): Convolutional filters, attention weights

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCompressionRatio(Int64,Int64)` | Calculates the compression ratio achieved. |
| `Compress(Vector<>)` | Compresses the given model weights. |
| `CompressMatrix(Matrix<>)` | Compresses a 2D matrix of weights (e.g., fully connected layer). |
| `CompressTensor(Tensor<>)` | Compresses an N-dimensional tensor of weights (e.g., convolutional filters). |
| `Decompress(Vector<>,ICompressionMetadata<>)` | Decompresses the compressed weights back to their original form. |
| `DecompressMatrix(Matrix<>,ICompressionMetadata<>)` | Decompresses the compressed matrix weights back to their original form. |
| `DecompressTensor(Tensor<>,ICompressionMetadata<>)` | Decompresses the compressed tensor weights back to their original form. |
| `GetCompressedSize(Matrix<>,ICompressionMetadata<>)` | Gets the size in bytes of the compressed matrix representation. |
| `GetCompressedSize(Tensor<>,ICompressionMetadata<>)` | Gets the size in bytes of the compressed tensor representation. |
| `GetCompressedSize(Vector<>,ICompressionMetadata<>)` | Gets the size in bytes of the compressed representation. |

