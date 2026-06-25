---
title: "QLoRAAdapter<T>"
description: "QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights.

## For Beginners

QLoRA is an advanced technique that makes fine-tuning large models
even more memory-efficient than standard LoRA. Here's how it works:

Imagine you have a huge model with millions of parameters:

- Standard LoRA: Freezes the base model, trains small adapters (huge memory savings)
- QLoRA: Does the same BUT also compresses the base model to 4-bit (even more savings!)

Think of it like storing a high-resolution image:

- Original model: Full 16-bit floating point (2 bytes per number)
- QLoRA base: Compressed to 4-bit (0.5 bytes per number)
- LoRA adapters: Still full precision (for accurate learning)

The result: You can fine-tune models 4x larger on the same hardware, or use 4x less GPU memory!

**When to use QLoRA vs Standard LoRA:**

- Use QLoRA when: GPU memory is very limited, model is huge, inference speed is critical
- Use Standard LoRA when: Memory is not a constraint, maximum accuracy is needed
- Both achieve similar quality in practice, QLoRA just uses less memory

**Trade-offs:**

- Pros: 75% less memory, same performance as 16-bit LoRA, faster inference after merging
- Cons: Slightly slower forward pass (dequantization overhead), more complex implementation

## How It Works

QLoRA extends the LoRA (Low-Rank Adaptation) technique by quantizing the base layer's weights
to 4-bit precision while keeping the LoRA adapter matrices (A and B) in full precision.
This achieves dramatic memory savings (typically 4x reduction) while maintaining training quality
comparable to full 16-bit fine-tuning.

**Key Features:**

- Base layer weights stored in 4-bit precision (INT4 or NF4)
- LoRA matrices (A and B) remain in full precision for accurate gradient updates
- Double quantization for constant quantization parameters (further memory savings)
- Paged optimizers support for handling memory spikes during training
- Dequantization happens on-the-fly during forward pass

**Memory Savings:**
For a typical transformer layer with 1000x1000 weights:

- Standard 16-bit: 2MB for weights
- QLoRA 4-bit base: 0.5MB for base weights + full precision LoRA (e.g., 32KB for rank 8)
- Total savings: ~75% memory reduction on base weights

**Quantization Types:**

- INT4: Uniform 4-bit integer quantization (-8 to 7)
- NF4 (4-bit Normal Float): Information-theoretically optimal for normally distributed weights

**Research Background:**
QLoRA was introduced in "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023).
It enables fine-tuning of 65B parameter models on a single 48GB GPU by combining:

1. 4-bit NormalFloat (NF4) quantization optimized for normally distributed weights
2. Double quantization to reduce memory footprint of quantization constants
3. Paged optimizers to handle memory spikes during gradient checkpointing

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new QLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured QLoRAAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QLoRAAdapter(ILayer<>,Int32,Double,QLoRAAdapter<>.QuantizationType,Boolean,Int32,Boolean)` | Initializes a new QLoRA adapter wrapping an existing Dense or FullyConnected layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BlockSize` | Gets the quantization block size. |
| `Quantization` | Gets the quantization type used for base layer weights. |
| `UsesDoubleQuantization` | Gets whether double quantization is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DequantizeINT4(Byte,,)` | Dequantizes a 4-bit integer value back to full precision. |
| `DequantizeNF4(Byte,,)` | Dequantizes a 4-bit Normal Float value back to full precision. |
| `DequantizeValue(Byte,,)` | Dequantizes a single 4-bit value back to full precision. |
| `DequantizeWeights` | Dequantizes the stored 4-bit weights back to full precision. |
| `DoubleQuantizeScales` | Applies double quantization to the scale factors to save additional memory. |
| `Forward(Tensor<>)` | Performs the forward pass through both quantized base and LoRA layers. |
| `MergeToOriginalLayer` | Merges the LoRA adaptation into the base layer and returns a quantized merged layer. |
| `QuantizeBaseLayerWeights` | Quantizes the base layer's weights to 4-bit precision. |
| `QuantizeINT4(,,)` | Quantizes a value using 4-bit integer quantization. |
| `QuantizeNF4(,,)` | Quantizes a value using 4-bit Normal Float quantization. |
| `QuantizeValue(,,)` | Quantizes a single value to 4-bit using the specified scale and zero point. |
| `ResetState` | Resets the internal state of the adapter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dequantizedWeights` | Cached dequantized weights for forward pass. |
| `_nf4Table` | NF4 quantization lookup table (16 values optimized for normal distribution). |
| `_quantizationBlockSize` | The block size for quantization (number of values sharing the same quantization parameters). |
| `_quantizationScales` | Scale factors for dequantization (one per quantization block). |
| `_quantizationType` | The type of quantization used for base layer weights. |
| `_quantizationZeroPoints` | Zero points for asymmetric quantization (one per quantization block). |
| `_quantizedWeights` | Quantized base layer weights stored as 4-bit values. |
| `_useDoubleQuantization` | Whether to use double quantization for quantization constants. |

