---
title: "InferenceOptimizationConfig"
description: "Configuration for inference-time optimizations to maximize prediction throughput and efficiency."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for inference-time optimizations to maximize prediction throughput and efficiency.

## For Beginners

Inference optimization makes your model's predictions faster and more efficient.

Key features:

- **KV Cache:** Remembers previous computations in attention layers (2-10x faster for long sequences)
- **Batching:** Groups multiple predictions together (higher throughput)
- **Speculative Decoding:** Uses a small model to draft tokens, then verifies (1.5-3x faster generation)

Default settings are optimized for most use cases. Simply enable and let the library handle the rest.

Example:

## How It Works

This configuration controls advanced inference optimizations including KV caching for transformers,
request batching for throughput, and speculative decoding for faster autoregressive generation.
These optimizations are automatically applied during prediction based on your configuration.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptiveBatchSize` | Gets or sets whether adaptive batch sizing is enabled. |
| `AttentionMasking` | Gets or sets how attention masking should be applied for optimized attention implementations. |
| `BatchTimeoutMs` | Gets or sets the maximum time to wait for batch to fill in milliseconds. |
| `Default` | Gets a default configuration with sensible settings for most use cases. |
| `DraftModelType` | Gets or sets the type of draft model to use for speculative decoding. |
| `EnableBatching` | Gets or sets whether request batching is enabled. |
| `EnableFlashAttention` | Gets or sets whether Flash Attention is enabled (when applicable). |
| `EnableKVCache` | Gets or sets whether KV (Key-Value) caching is enabled for attention layers. |
| `EnableLayerFusion` | Gets or sets whether freeze-time layer fusion (BatchNorm folding) is applied when optimizing a model for inference. |
| `EnablePagedKVCache` | Gets or sets whether to use a paged KV-cache backend (vLLM-style) for long-context / multi-sequence serving. |
| `EnableSpeculativeDecoding` | Gets or sets whether speculative decoding is enabled. |
| `EnableWeightOnlyQuantization` | Gets or sets whether weight-only INT8 quantization is enabled for inference. |
| `HighPerformance` | Gets a high-performance configuration optimized for maximum throughput. |
| `InferenceQuantization` | Gets or sets the weight quantization mode used for inference. |
| `KVCacheEvictionPolicy` | Gets or sets the KV cache eviction policy. |
| `KVCacheMaxSizeMB` | Gets or sets the maximum KV cache size in megabytes. |
| `KVCachePrecision` | Gets or sets the precision used for KV-cache storage. |
| `KVCacheQuantization` | Gets or sets the quantization mode used for KV-cache storage. |
| `KVCacheWindowSize` | Gets or sets the sliding window size in tokens when `UseSlidingWindowKVCache` is enabled. |
| `MaxBatchSize` | Gets or sets the maximum batch size for grouped predictions. |
| `MinBatchSize` | Gets or sets the minimum batch size before processing. |
| `PagedKVCacheBlockSize` | Gets or sets the block size (in tokens) for the paged KV-cache when enabled. |
| `PositionalEncoding` | Gets or sets the positional encoding type for attention layers. |
| `RoPETheta` | Gets or sets the base frequency parameter for RoPE (theta). |
| `SpeculationDepth` | Gets or sets the speculation depth (number of tokens to draft ahead). |
| `SpeculationPolicy` | Gets or sets the policy for when speculative decoding should run. |
| `SpeculativeMethod` | Gets or sets the speculative decoding method. |
| `UseSlidingWindowKVCache` | Gets or sets whether to use a sliding window KV-cache for long contexts. |
| `UseTreeSpeculation` | Gets or sets whether to use tree-structured speculation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the configuration and throws if any values are invalid. |

