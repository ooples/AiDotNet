---
title: "LoftQAdapter<T>"
description: "LoftQ (LoRA-Fine-Tuning-Quantized) adapter that combines quantization and LoRA with improved initialization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

LoftQ (LoRA-Fine-Tuning-Quantized) adapter that combines quantization and LoRA with improved initialization.

## For Beginners

LoftQ is an improved version of QLoRA that starts with better settings.

Think of it like this:

- QLoRA: Compress your model, then add random corrections, then train
- LoftQ: Compress your model, figure out what corrections are needed upfront, then train

The key insight: If we're going to compress the weights anyway, let's make sure our
correction layer (LoRA) is specifically designed to fix compression errors!

The process:

1. Start with your pre-trained model
2. Repeatedly:
- Try different compressions
- Adjust LoRA to compensate for compression error
- Pick the best combination
3. Now train LoRA (which already knows how to fix compression issues)

Benefits:

- Better starting point for training
- Converges faster during fine-tuning
- Better final accuracy than QLoRA with same memory usage
- Still only trains LoRA (same efficiency as QLoRA)

Trade-offs:

- Longer initialization time (worth it for better results)
- Same runtime memory and speed as QLoRA
- More complex implementation

## How It Works

LoftQ improves upon QLoRA by using an alternating optimization strategy during initialization
to find better LoRA adapter parameters for quantized models. Instead of simply quantizing
a pre-trained model and adding LoRA on top, LoftQ alternates between:

1. Optimizing the quantization of the base weights
2. Optimizing the LoRA adapter matrices to compensate for quantization error

**Key Features:**

- Alternating optimization between quantization and LoRA initialization
- Better initialization than naive quantization + LoRA
- Supports both 4-bit INT4 and NF4 quantization
- Reduces the gap between quantized and full-precision fine-tuning
- Compatible with all QLoRA features (double quantization, block-wise quantization)

**How LoftQ Differs from QLoRA:**
QLoRA:

1. Quantize pre-trained weights
2. Initialize LoRA randomly
3. Fine-tune LoRA only

LoftQ:

1. Start with pre-trained weights
2. Alternate K times:

a. Fix LoRA, optimize quantization
b. Fix quantization, optimize LoRA (via SVD to minimize error)

3. Fine-tune LoRA only

This alternating initialization creates better starting LoRA parameters that compensate
for quantization error from the beginning, leading to better final performance.

**Alternating Optimization Process:**
For K iterations (typically 3-5):

- Quantization step: Quantize W to get Q, keeping A and B fixed
- LoRA step: Update A and B to minimize ||W - (Q + AB)||, keeping Q fixed

This ensures the LoRA adapter specifically compensates for quantization error,
rather than learning generic adaptations.

**Memory Efficiency:**
Same as QLoRA - base weights in 4-bit, LoRA in full precision:

- 75% memory reduction on base weights
- Only LoRA parameters trainable (typically 0.1-1% of model size)
- Additional one-time cost during initialization for alternating optimization

**Research Background:**
LoftQ was introduced in "LoftQ: LoRA-Fine-Tuning-Aware Quantization" (Li et al., 2023).
It addresses a key limitation of QLoRA: random LoRA initialization doesn't account for
the specific quantization errors introduced. By using alternating optimization, LoftQ
creates LoRA parameters that are "aware" of the quantization, leading to better downstream
fine-tuning performance with no additional runtime cost.

**When to Use LoftQ vs QLoRA:**

- Use LoftQ when: Training accuracy is critical, willing to spend extra time on initialization
- Use QLoRA when: Fast experimentation needed, initialization time is critical
- Both have identical runtime memory and speed characteristics

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new LoftQAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured LoftQAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoftQAdapter(ILayer<>,Int32,Double,Int32,LoftQAdapter<>.QuantizationType,Boolean,Int32,Boolean)` | Initializes a new LoftQ adapter with alternating optimization for improved initialization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlternatingIterations` | Gets the number of alternating optimization iterations used during initialization. |
| `BlockSize` | Gets the quantization block size. |
| `Quantization` | Gets the quantization type used for base layer weights. |
| `UsesDoubleQuantization` | Gets whether double quantization is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DequantizeINT4(Byte,,)` | Dequantizes a 4-bit integer value. |
| `DequantizeNF4(Byte,,)` | Dequantizes a 4-bit Normal Float value. |
| `DequantizeValue(Byte,,)` | Dequantizes a single 4-bit value. |
| `DequantizeWeights` | Dequantizes the stored 4-bit weights back to full precision. |
| `DoubleQuantizeScales` | Applies double quantization to scale factors. |
| `Forward(Tensor<>)` | Performs the forward pass through quantized base layer and LoRA. |
| `MergeToOriginalLayer` | Merges LoRA adaptation into base layer and returns merged layer. |
| `PerformLoftQInitialization` | Performs LoftQ initialization using alternating optimization between quantization and LoRA. |
| `QuantizeINT4(,,)` | Quantizes a value using 4-bit integer quantization. |
| `QuantizeNF4(,,)` | Quantizes a value using 4-bit Normal Float quantization. |
| `QuantizeValue(,,)` | Quantizes a single value to 4-bit. |
| `QuantizeWeights(Matrix<>)` | Quantizes a weight matrix to 4-bit precision. |
| `ResetState` | Resets the internal state of the adapter. |
| `UpdateLoRAFromResidual(Matrix<>)` | Updates LoRA matrices A and B to minimize the residual via SVD decomposition. |
| `UpdateParametersFromLayers` | Updates the parameter vector from both layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dequantizedWeights` | Cached dequantized weights for forward pass. |
| `_nf4Table` | NF4 quantization lookup table (16 values optimized for normal distribution). |
| `_numAlternatingIterations` | Number of alternating optimization iterations during initialization. |
| `_quantizationBlockSize` | The block size for quantization. |
| `_quantizationScales` | Scale factors for dequantization (one per quantization block). |
| `_quantizationType` | The type of quantization used for base layer weights. |
| `_quantizationZeroPoints` | Zero points for asymmetric quantization (one per quantization block). |
| `_quantizedWeights` | Quantized base layer weights stored as 4-bit values. |
| `_useDoubleQuantization` | Whether to use double quantization for quantization constants. |

