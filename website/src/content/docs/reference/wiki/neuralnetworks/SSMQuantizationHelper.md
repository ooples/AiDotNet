---
title: "SSMQuantizationHelper<T>"
description: "Provides SSM-specific quantization utilities for reducing memory and accelerating inference."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Provides SSM-specific quantization utilities for reducing memory and accelerating inference.

## For Beginners

Quantization makes the model smaller and faster by using less precise numbers.

Think of it like rounding numbers:

- Full precision: 3.14159265 (32-bit float, 4 bytes per number)
- INT8 quantized: 3.14 (8-bit integer with scale, 1 byte per number)
- The model gets 4x smaller and often runs 2-4x faster

For SSM models specifically:

- The A, B, C weight matrices benefit the most from quantization (biggest savings)
- The D parameter (skip connection) is sensitive - we protect it
- Hidden states can also be quantized during generation to save memory

This class provides tools to quantize SSM layers intelligently, knowing which parts
are safe to compress and which need protection.

## How It Works

SSM models like Mamba have unique quantization characteristics compared to Transformers:

- The A, B, C projection weights are the most impactful for compression (they dominate parameter count)
- The D skip connection parameter is very sensitive and should usually be kept in full precision
- Hidden states can be quantized during inference for memory-constrained deployment
- The Conv1D weights are small and benefit less from quantization

This utility works with the existing `QuantizationConfiguration` and
`IQuantizer` infrastructure to provide SSM-aware quantization.
It applies quantization at the parameter level (via GetParameters/SetParameters) rather than
creating wrapped layer types, keeping the architecture clean.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeQuantizationError(ILayer<>,Int32)` | Computes the quantization error (mean absolute error) that would result from quantizing a layer. |
| `EstimateMemorySavings(ILayer<>,Int32)` | Estimates the memory savings from quantizing an SSM layer at the specified bit width. |
| `QuantizeSSMLayer(ILayer<>,QuantizationConfiguration,Boolean)` | Quantizes an SSM layer's parameters using the provided configuration. |
| `QuantizeStateCache(SSMStateCache<>,Int32)` | Creates a new SSM state cache with precision-reduced states migrated from the source cache. |

