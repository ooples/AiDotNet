---
title: "QALoRAAdapter<T>"
description: "Quantization-Aware LoRA (QA-LoRA) adapter that combines parameter-efficient fine-tuning with group-wise quantization awareness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

Quantization-Aware LoRA (QA-LoRA) adapter that combines parameter-efficient fine-tuning with group-wise quantization awareness.

## For Beginners

QA-LoRA solves a critical problem when deploying models to resource-constrained devices.

The Problem:

- Modern neural networks use high-precision numbers (32-bit floats)
- Mobile/edge devices need lower precision (4-bit or 8-bit integers) for speed and memory
- Converting after training (post-training quantization) often loses accuracy

QA-LoRA's Solution:

- Simulates low-precision during training (quantization-aware training)
- Learns to compensate for quantization errors
- Uses LoRA for parameter efficiency (only trains the adaptation, not full model)
- Applies group-wise quantization (groups of weights share scaling factors)

Key Concepts:

1. Quantization: Converting high-precision numbers to low-precision

Example: 32-bit float 0.7234 → 4-bit integer 11 (range 0-15)

2. Group-wise Quantization: Instead of one scale for all weights, weights are divided into groups,

each with its own scale. This preserves more information.
Example: 64 weights → 4 groups of 16 weights each, each group has its own scale

3. Quantization-Aware Training: During training, simulate quantization in forward pass:
- Convert weights to low-precision (quantize)
- Immediately convert back to high-precision (dequantize)
- Use these "quantized" values for computation
- Gradients learn to compensate for the quantization noise

4. Straight-Through Estimator (STE): During backward pass, treat quantization as identity
- Forward: y = quantize(x)
- Backward: ∂y/∂x ≈ 1 (gradient flows through unchanged)
- This allows gradients to update the full-precision weights

Parameters:

- QuantizationBits: How many bits to use (4-bit, 8-bit, etc.)
- GroupSize: How many weights per quantization group (e.g., 64, 128)
- Smaller GroupSize = more scales = better accuracy but more overhead
- Larger GroupSize = fewer scales = more efficient but less accurate

Example Workflow:

1. Training: Forward pass uses simulated 4-bit quantization
2. Gradients: Backward pass learns to work around quantization errors
3. Deployment: Actually quantize the merged weights to 4-bit for inference
4. Result: Much better accuracy than quantizing after training

Research Context:

- QLoRA (May 2023): Introduced efficient 4-bit quantization for LoRA
- QA-LoRA: Extends this with quantization-aware training for better results
- Typical improvement: 1-3% accuracy gain over post-training quantization

Use Cases:

- Deploying large language models on mobile devices
- Edge AI applications with strict memory constraints
- Reducing model size while maintaining accuracy
- Fine-tuning for deployment on specific hardware (TPUs, specialized accelerators)

## How It Works

QA-LoRA extends standard LoRA by being aware of quantization during training. This allows the adapter
to learn compensations for quantization errors, resulting in better final accuracy compared to
post-training quantization approaches. The key innovation is simulating quantization during the
forward pass so that gradients account for quantization effects.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QALoRAAdapter(ILayer<>,Int32,Int32,Int32,Double,Boolean)` | Initializes a new QA-LoRA adapter with quantization awareness. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GroupSize` | Gets or sets the group size for group-wise quantization. |
| `QuantizationBits` | Gets or sets the number of bits used for quantization. |
| `QuantizationEnabled` | Gets or sets whether quantization simulation is enabled during forward/backward passes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through both base and LoRA layers with quantization simulation. |
| `GetQuantizationStats` | Gets statistics about quantization for the current LoRA parameters. |
| `MergeToOriginalLayer` | Merges the LoRA adaptation into the base layer and returns a quantized merged layer. |
| `QuantizeAndDequantize(Vector<>)` | Simulates quantization and dequantization using group-wise scaling. |
| `UpdateParameterGradientsFromLayers` | Updates parameter gradients from both layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_groupSize` | Number of weights per quantization group. |
| `_quantizationBits` | Number of bits to use for quantization (e.g., 4, 8). |
| `_quantizationEnabled` | Whether quantization simulation is currently enabled. |

