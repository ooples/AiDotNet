---
title: "TensorCompressionMetadata<T>"
description: "Metadata for N-dimensional tensor compression operations that wraps the underlying vector compression metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for N-dimensional tensor compression operations that wraps the underlying vector compression metadata.

## For Beginners

Tensors are multi-dimensional arrays used extensively in deep learning:

- 1D tensor (vector): [100] - like a bias term with 100 values
- 2D tensor (matrix): [100, 50] - like fully connected layer weights
- 3D tensor: [32, 100, 50] - like a batch of 32 matrices
- 4D tensor: [64, 3, 3, 3] - like 64 convolutional filters with 3 channels, 3x3 kernels

When compressing a tensor, we flatten it to a 1D array, compress it, and need to remember
the original shape to reconstruct it. This metadata stores:

1. The original shape - the dimensions of the tensor (e.g., [64, 3, 3, 3])
2. The inner compression details - how the flattened data was compressed

Think of it like packing a Rubik's cube for shipping:

- You disassemble it into individual pieces (flatten)
- You put the pieces in a compressed bag (apply compression)
- You include assembly instructions with the dimensions (this metadata)
- Later, you can perfectly reconstruct the original cube

## How It Works

TensorCompressionMetadata stores the information needed to decompress an N-dimensional weight tensor that was
compressed by first flattening it to a vector. It preserves the original tensor shape (dimensions) and
delegates the actual compression metadata to an inner ICompressionMetadata instance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorCompressionMetadata(Int32[],ICompressionMetadata<>)` | Initializes a new instance of the TensorCompressionMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InnerMetadata` | Gets the inner compression metadata from the underlying vector compression algorithm. |
| `OriginalLength` | Gets the original total number of elements in the flattened tensor. |
| `OriginalShape` | Gets a copy of the original shape (dimensions) of the tensor. |
| `Rank` | Gets the number of dimensions in the original tensor. |
| `Type` | Gets the compression type from the underlying compression algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the total size in bytes of this metadata structure, including the inner metadata. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_originalShape` | The internal backing field for OriginalShape, kept private to ensure immutability. |

