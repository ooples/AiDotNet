---
title: "DeepCompressionStats"
description: "Statistics about Deep Compression performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Statistics about Deep Compression performance.

## For Beginners

These statistics help you understand how well compression worked:

- CompressionRatio: How much smaller the model is (e.g., 35x means 35 times smaller)
- Sparsity: What fraction of weights are zero (e.g., 0.9 = 90% zeros)
- BitsPerWeight: How many bits are used per non-zero weight

## Properties

| Property | Summary |
|:-----|:--------|
| `BitsPerWeight` | Effective bits per weight after quantization. |
| `CompressedSizeBytes` | Compressed size in bytes after all three stages. |
| `CompressionRatio` | Overall compression ratio (original / compressed). |
| `NumClusters` | Number of clusters used for quantization. |
| `OriginalSizeBytes` | Original size in bytes before compression. |
| `PruningRatio` | Compression ratio from pruning stage alone. |
| `QuantizationRatio` | Compression ratio from quantization stage alone. |
| `Sparsity` | Sparsity achieved by pruning (fraction of zeros). |

