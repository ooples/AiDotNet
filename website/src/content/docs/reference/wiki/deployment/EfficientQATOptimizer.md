---
title: "EfficientQATOptimizer<T>"
description: "EfficientQAT optimizer providing memory-efficient Quantization-Aware Training for large models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Training`

EfficientQAT optimizer providing memory-efficient Quantization-Aware Training for large models.
Uses block-wise quantization and efficient gradient computation to reduce memory footprint.

## For Beginners

Standard QAT uses a lot of memory because it keeps full-precision
copies of all weights. EfficientQAT is smarter about memory, letting you train bigger models
on the same hardware.

## How It Works

**Key Innovations:**

**Memory Savings:** 2-4x less memory than standard QAT

**Reference:** "Efficient Quantization-Aware Training for Large Language Models" (ACL 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EfficientQATOptimizer(QuantizationConfiguration,Int32)` | Initializes a new instance of the EfficientQATOptimizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentBitWidth` | Gets the current effective bit width (may change with progressive quantization). |
| `QATHook` | Gets the QAT training hook for applying fake quantization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBlockWiseQuantization(Vector<>,String)` | Applies block-wise fake quantization to weights with memory efficiency. |
| `ComputeQuantizationAwareGradient(Vector<>,Vector<>,String)` | Computes gradients with block-wise quantization awareness. |
| `ComputeQuantizationParameters(Double,Double,Double,Int32,Double,Boolean)` | Computes quantization parameters (scale and zero-point) for a given value range. |
| `EstimateMemoryUsage(Int64)` | Gets memory usage estimate for current configuration. |
| `InitializeBlockState(Vector<>,String,Int32)` | Initializes block quantization state for a layer. |
| `OnEpochStart(Int32)` | Called at the start of each epoch to manage progressive quantization. |
| `UpdateBlockScales(Vector<>,String,Double)` | Updates block scales based on observed weight distributions. |
| `UpdateProgressiveQuantization` | Updates progressive quantization schedule. |

