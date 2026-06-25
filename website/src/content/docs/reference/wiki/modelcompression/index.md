---
title: "Model Compression"
description: "All 26 public types in the AiDotNet.modelcompression namespace, organized by kind."
section: "API Reference"
---

**26** public types in this namespace, organized by kind.

## Models & Types (23)

| Type | Summary |
|:-----|:--------|
| [`CompressionAnalyzer<T>`](/docs/reference/wiki/modelcompression/compressionanalyzer/) | Analyzes model weights to determine optimal compression strategies. |
| [`CompressionMetrics<T>`](/docs/reference/wiki/modelcompression/compressionmetrics/) | Provides metrics and statistics for model compression operations. |
| [`CompressionResult<T>`](/docs/reference/wiki/modelcompression/compressionresult/) | Result of a compression operation. |
| [`DeepCompressionMetadata<T>`](/docs/reference/wiki/modelcompression/deepcompressionmetadata/) | Metadata for Deep Compression containing information from all three stages. |
| [`DeepCompressionStats`](/docs/reference/wiki/modelcompression/deepcompressionstats/) | Statistics about Deep Compression performance. |
| [`DeepCompression<T>`](/docs/reference/wiki/modelcompression/deepcompression/) | Implements the Deep Compression algorithm from Han et al. |
| [`HuffmanEncodingCompression<T>`](/docs/reference/wiki/modelcompression/huffmanencodingcompression/) | Implements Huffman encoding compression for model weights using variable-length encoding. |
| [`HuffmanEncodingMetadata<T>`](/docs/reference/wiki/modelcompression/huffmanencodingmetadata/) | Metadata for Huffman encoding compression. |
| [`HybridCompressionMetadata`](/docs/reference/wiki/modelcompression/hybridcompressionmetadata/) | Legacy non-generic metadata for backward compatibility. |
| [`HybridCompressionMetadata<T>`](/docs/reference/wiki/modelcompression/hybridcompressionmetadata-2/) | Metadata for hybrid compression combining clustering and Huffman encoding. |
| [`HybridHuffmanClusteringCompression<T>`](/docs/reference/wiki/modelcompression/hybridhuffmanclusteringcompression/) |  |
| [`LowRankFactorizationCompression<T>`](/docs/reference/wiki/modelcompression/lowrankfactorizationcompression/) | Implements Low-Rank Factorization compression using SVD-like decomposition. |
| [`LowRankFactorizationMetadata<T>`](/docs/reference/wiki/modelcompression/lowrankfactorizationmetadata/) | Metadata for Low-Rank Factorization compression. |
| [`MatrixCompressionMetadata<T>`](/docs/reference/wiki/modelcompression/matrixcompressionmetadata/) | Metadata for matrix compression operations that wraps the underlying vector compression metadata. |
| [`ProductQuantizationCompression<T>`](/docs/reference/wiki/modelcompression/productquantizationcompression/) | Implements Product Quantization (PQ) compression for model weights. |
| [`ProductQuantizationMetadata<T>`](/docs/reference/wiki/modelcompression/productquantizationmetadata/) | Metadata for Product Quantization compression. |
| [`SparseCompressionResult<T>`](/docs/reference/wiki/modelcompression/sparsecompressionresult/) | Result of sparse compression operation. |
| [`SparsePruningCompression<T>`](/docs/reference/wiki/modelcompression/sparsepruningcompression/) | Implements sparse pruning compression by zeroing out small-magnitude weights. |
| [`SparsePruningMetadata<T>`](/docs/reference/wiki/modelcompression/sparsepruningmetadata/) | Metadata for sparse pruning compression. |
| [`TensorCompressionMetadata<T>`](/docs/reference/wiki/modelcompression/tensorcompressionmetadata/) | Metadata for N-dimensional tensor compression operations that wraps the underlying vector compression metadata. |
| [`WeightAnalysisResult<T>`](/docs/reference/wiki/modelcompression/weightanalysisresult/) | Analysis results for model weights to guide compression decisions. |
| [`WeightClusteringCompression<T>`](/docs/reference/wiki/modelcompression/weightclusteringcompression/) | Implements weight clustering compression using K-means clustering to group similar weights. |
| [`WeightClusteringMetadata<T>`](/docs/reference/wiki/modelcompression/weightclusteringmetadata/) | Metadata for weight clustering compression. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`ModelCompressionBase<T>`](/docs/reference/wiki/modelcompression/modelcompressionbase/) | Provides a base implementation for model compression techniques used to reduce model size while preserving accuracy. |

## Enums (1)

| Type | Summary |
|:-----|:--------|
| [`SparseFormat`](/docs/reference/wiki/modelcompression/sparseformat/) | Sparse storage formats. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`HuffmanNode<T>`](/docs/reference/wiki/modelcompression/huffmannode/) | Represents a node in the Huffman tree. |

