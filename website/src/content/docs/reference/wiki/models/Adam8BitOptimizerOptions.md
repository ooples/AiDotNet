---
title: "Adam8BitOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the 8-bit Adam optimization algorithm, which reduces memory usage by quantizing optimizer states."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the 8-bit Adam optimization algorithm, which reduces memory usage by quantizing optimizer states.

## For Beginners

Training large neural networks requires storing "optimizer state" - extra numbers
for each parameter that help the optimizer make better updates. Standard Adam stores two numbers per parameter
(momentum and variance), which can use a lot of memory for large models.

8-bit Adam compresses these numbers using a technique called quantization, similar to how JPEG compresses images.
This reduces memory usage significantly with minimal impact on training quality. It's especially useful when
training large models where optimizer memory becomes a bottleneck.

## How It Works

8-bit Adam provides the same optimization behavior as standard Adam but stores the first and second moment
estimates (m and v) using 8-bit quantized representations instead of full precision floating point.
This reduces memory usage by approximately 4x for these optimizer states, which is significant for large models.

**Memory Savings Example:**
For a model with 1 billion parameters:

- Standard Adam: 8 GB for optimizer states (2 states × 4 bytes × 1B params)
- 8-bit Adam: ~2 GB for optimizer states (2 states × 1 byte × 1B params + scaling factors)

## Properties

| Property | Summary |
|:-----|:--------|
| `BlockSize` | Gets or sets the block size for block-wise quantization. |
| `CompressBothMoments` | Gets or sets whether to compress both first and second moments. |
| `FullPrecisionUpdateFrequency` | Gets or sets the frequency of full-precision state updates. |
| `QuantizationPercentile` | Gets or sets the percentile to use for outlier-aware quantization. |
| `UseBFloat16MomentStorage` | Stores the optimizer moment state (m and v) as BFloat16 (2 bytes/element) instead of the default 8-bit block-quantized representation (1 byte/element). |
| `UseDynamicQuantization` | Gets or sets whether to use dynamic quantization that adapts the scale during training. |
| `UseStochasticRounding` | Gets or sets whether to use stochastic rounding during quantization. |

