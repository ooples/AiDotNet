---
title: "FlashAttentionConfig"
description: "Configuration options for Flash Attention algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NeuralNetworks.Attention`

Configuration options for Flash Attention algorithm.

## For Beginners

Flash Attention is a faster way to compute attention.

Standard attention creates a huge matrix comparing every position to every other position.
For long sequences (like 4096 tokens), this matrix has 16 million entries!

Flash Attention avoids creating this huge matrix by:

- Processing in small blocks that fit in fast GPU memory (SRAM)
- Computing softmax incrementally as it processes each block
- Never storing the full attention matrix

Benefits:

- 2-4x faster than standard attention
- Uses much less memory (O(N) instead of O(N^2))
- Enables training with longer sequences

## How It Works

Flash Attention is a memory-efficient attention algorithm that avoids materializing
the full N x N attention matrix. Instead, it processes attention in tiles/blocks,
computing online softmax incrementally.

## Properties

| Property | Summary |
|:-----|:--------|
| `BlockSizeKV` | Block size for key/value processing (Bc in the paper). |
| `BlockSizeQ` | Block size for query processing (Br in the paper). |
| `Causal` | Creates a configuration optimized for causal/autoregressive models. |
| `Default` | Creates a default configuration suitable for most use cases. |
| `DropoutProbability` | Dropout probability to apply to attention weights during training. |
| `HighPerformance` | Creates a configuration optimized for speed (uses more memory). |
| `MemoryEfficient` | Creates a configuration optimized for memory efficiency. |
| `Precision` | Numerical precision mode for attention computation. |
| `RecomputeInBackward` | Whether to enable memory-efficient backward pass with recomputation. |
| `ReturnAttentionWeights` | Whether to return attention weights (for visualization/debugging). |
| `ScaleFactor` | Scale factor for attention scores. |
| `UseCausalMask` | Whether to apply causal masking (for autoregressive models). |
| `UseGpuKernel` | Whether to use the optimized GPU kernel (when available). |

