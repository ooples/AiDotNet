---
title: "ProductQuantizationMetadata<T>"
description: "Metadata for Product Quantization compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for Product Quantization compression.

## For Beginners

This metadata stores:

- Codebooks: The representative values for each subvector position
- Dimensions: How the original vector was divided
- Original length: For proper reconstruction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProductQuantizationMetadata([][],Int32,Int32,Int32,Int32)` | Initializes a new instance of the ProductQuantizationMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Codebooks` | Gets the codebooks for each subvector position. |
| `NumCentroids` | Gets the number of centroids per codebook. |
| `NumSubvectors` | Gets the number of subvectors. |
| `OriginalLength` | Gets the original length of the weights array. |
| `SubvectorDimension` | Gets the dimension of each subvector. |
| `Type` | Gets the compression type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

