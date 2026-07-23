# QLoRA Fine-Tuning - 4-bit Quantized LoRA

This sample demonstrates QLoRA (Quantized LoRA) for ultra-memory-efficient fine-tuning using AiDotNet, enabling training of large models on consumer hardware through 4-bit quantization.

## What You'll Learn

- How QLoRA achieves 75% memory reduction through 4-bit quantization
- How NF4 (4-bit Normal Float) optimizes for neural network weights
- How to compare memory usage between full fine-tuning, LoRA, and QLoRA
- How to configure and use QLoRA adapters
- Real-world GPU requirements for different model sizes

## What is QLoRA?

QLoRA combines 4-bit quantization with LoRA for maximum efficiency:

```
Standard LoRA:
  Base weights: FP16 (2 bytes per parameter)
  LoRA adapters: FP32 (4 bytes per parameter)
  Total: ~2 bytes per base parameter + LoRA overhead

QLoRA:
  Base weights: 4-bit (0.5 bytes per parameter)
  LoRA adapters: FP32 (4 bytes per parameter)
  Total: ~0.5 bytes per base parameter + LoRA overhead

Memory Reduction: ~75% on base model weights!
```

## Running the Sample

```bash
cd samples/llm-fine-tuning/QLoRA
dotnet run
```

## Expected Output

```
=== AiDotNet QLoRA Fine-Tuning ===
4-bit Quantized LoRA for Memory-Efficient Fine-Tuning

Configuration:
  - Input Size: 256
  - Hidden Size: 512
  - Output Size: 10
  - LoRA Rank: 8
  - LoRA Alpha: 8.0

Creating base model layers...

======================================================================
Memory Analysis: Standard vs QLoRA
======================================================================

Base Model Memory (without LoRA):
  Total Parameters: 398,858
  Memory (FP32): 1.52 MB
  Memory (FP16): 778.04 KB

QLoRA Memory Breakdown:
  Base weights (4-bit): 194.76 KB
  Quantization constants: 5.84 KB
  LoRA adapters (FP32): 50.94 KB
  Total QLoRA memory: 251.54 KB

Memory Savings vs FP16: 67.7% (3.1x reduction)

======================================================================
QLoRA Quantization Configuration
======================================================================

Quantization Types:
  - INT4: Uniform 4-bit integer (-8 to 7)
  - NF4: 4-bit Normal Float (optimal for normally distributed weights)

NF4 Quantization Levels (16 values optimized for normal distribution):
  Index | NF4 Value | Description
  ------|-----------|-------------
      0 |   -1.0000 | Minimum
      1 |   -0.6962 |
      2 |   -0.5251 |
      ...
      7 |    0.0000 | Zero
      ...
     15 |    1.0000 | Maximum

Notice: Values are NOT evenly spaced - more resolution near zero
        where most neural network weights are concentrated.

======================================================================
Creating QLoRA Adapters
======================================================================

Initializing QLoRA adapter for Layer 1...
  - Input size: 256
  - Output size: 512
  - LoRA rank: 8
  - Quantization: NF4
  - Double quantization: Enabled
  - Block size: 64
  [OK] Layer 1 QLoRA adapter created

QLoRA Adapter Properties:
  Quantization Type: NF4
  Double Quantization: True
  Block Size: 64

======================================================================
Memory Comparison Visualization
======================================================================

  Full FP32 Fine-Tuning      [########################################] 1.52 MB
                             Trainable: 398,858 params

  Full FP16 Fine-Tuning      [####################....................] 778.04 KB
                             Trainable: 398,858 params

  Standard LoRA (FP16 base)  [####################....................] 828.98 KB
                             Trainable: 13,064 params

  QLoRA (4-bit base)         [######..................................] 251.54 KB
                             Trainable: 13,064 params

======================================================================
QLoRA vs Standard LoRA Comparison
======================================================================

| Aspect                  | Standard LoRA    | QLoRA            |
|-------------------------|------------------|------------------|
| Base weight precision   | FP16 (2 bytes)   | INT4 (0.5 bytes) |
| LoRA adapter precision  | FP32             | FP32             |
| Memory for base         | 778.04 KB        | 194.76 KB        |
| Memory for LoRA         | 50.94 KB         | 50.94 KB         |
| Total memory            | 828.98 KB        | 251.54 KB        |
| Trainable parameters    | 13,064           | 13,064           |
| Forward pass overhead   | None             | Dequantization   |
| Best for                | Speed priority   | Memory limited   |

======================================================================
Real-World Scaling: 7B Parameter Model Example
======================================================================

7B Parameter Model Memory Requirements:
  Full Fine-Tuning (FP32):    26.08 GB (need ~56GB GPU)
  Full Fine-Tuning (FP16):    13.04 GB (need ~28GB GPU)
  Standard LoRA (FP16 base):  13.30 GB (need ~28GB GPU)
  QLoRA (4-bit base):         3.52 GB (need ~7GB GPU)

With QLoRA, you can fine-tune a 7B model on a consumer GPU!
  - NVIDIA RTX 3080 (10GB): Can train 7B models
  - NVIDIA RTX 4090 (24GB): Can train 13B-30B models
  - Single A100 (80GB): Can train 65B+ models
```

## How QLoRA Works

### 1. 4-bit NF4 Quantization

NF4 (4-bit Normal Float) is optimized for normally distributed neural network weights:

```
Standard INT4: Equal spacing
  [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

NF4: Unequal spacing (more values near zero)
  [-1.0, -0.70, -0.53, -0.39, -0.28, -0.18, -0.09, 0.0,
    0.08,  0.16,  0.25,  0.34,  0.44,  0.56,  0.72, 1.0]
```

Neural network weights follow a normal distribution centered at zero, so NF4 preserves more precision where it matters most.

### 2. Double Quantization

QLoRA quantizes the quantization constants (scales and zero-points) themselves:

```
Without Double Quantization:
  Weights: 4-bit
  Scales: FP32 (one per block of 64 weights)
  Overhead: ~6% for scales

With Double Quantization:
  Weights: 4-bit
  Scales: 8-bit (quantized)
  Overhead: ~3% for scales
```

### 3. Block-wise Quantization

Weights are quantized in blocks for better accuracy:

```
Block Size: 64 (default)
  - Each block of 64 weights shares one scale and zero-point
  - Smaller blocks = better accuracy, more overhead
  - Larger blocks = worse accuracy, less overhead
```

## Code Highlights

### Creating QLoRA Adapter

```csharp
var qloraLayer = new QLoRAAdapter<double>(
    baseLayer: denseLayer,           // Original layer to adapt
    rank: 8,                         // LoRA rank
    alpha: 8.0,                      // LoRA scaling
    quantizationType: QLoRAAdapter<double>.QuantizationType.NF4,  // NF4 recommended
    useDoubleQuantization: true,     // Additional memory savings
    quantizationBlockSize: 64,       // Block size for quantization
    freezeBaseLayer: true            // Freeze base weights
);
```

### Forward Pass (with Dequantization)

```csharp
// Internally, QLoRA:
// 1. Dequantizes base weights (4-bit -> FP32)
// 2. Computes base layer output
// 3. Computes LoRA output (full precision)
// 4. Sums the outputs

var output = qloraLayer.Forward(input);
```

### Backward Pass (LoRA Only)

```csharp
// Only LoRA parameters receive gradients
// Base weights remain frozen and quantized

var gradient = qloraLayer.Backward(outputGradient);
```

### Merging for Deployment

```csharp
// After training, merge LoRA into base weights
var mergedLayer = qloraLayer.MergeToOriginalLayer();

// Result can be:
// - Re-quantized to 4-bit for efficient inference
// - Kept in FP16/FP32 for maximum accuracy
```

## Memory Savings Table

| Model Size | FP32 | FP16 | LoRA (FP16) | QLoRA (4-bit) |
|------------|------|------|-------------|---------------|
| 7B | 28 GB | 14 GB | 14.3 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 26.5 GB | 6.5 GB |
| 30B | 120 GB | 60 GB | 61 GB | 15 GB |
| 65B | 260 GB | 130 GB | 132 GB | 33 GB |

## GPU Requirements

| GPU | VRAM | Max Model with QLoRA |
|-----|------|----------------------|
| RTX 3060 | 12 GB | 7B |
| RTX 3080 | 10 GB | 7B |
| RTX 3090 | 24 GB | 13B |
| RTX 4090 | 24 GB | 13B-30B |
| A100 | 40/80 GB | 30B-65B |

## Architecture

```
                Input
                  |
                  v
    +---------------------------+
    | QLoRA Forward Pass        |
    |                           |
    |  1. Dequantize base       |
    |     (4-bit -> FP32)       |
    |                           |
    |  2. Base output           |
    |     W_dequant * x         |
    |                           |
    |  3. LoRA output (FP32)    |
    |     B * A * x * (a/r)     |
    |                           |
    |  4. Sum outputs           |
    +------------+--------------+
                 |
                 v
              Output
```

## Trade-offs

| Aspect | Pro | Con |
|--------|-----|-----|
| Memory | 75% reduction | - |
| Speed | - | Dequantization overhead |
| Accuracy | Near full precision | Slight quantization loss |
| Complexity | - | More hyperparameters |

## When to Use QLoRA

**Use QLoRA when:**
- GPU memory is limited
- Training very large models (7B+)
- Fine-tuning on consumer hardware
- Multiple adapters needed (memory per adapter is small)

**Use Standard LoRA when:**
- Inference speed is critical
- Training on high-memory GPUs
- Maximum accuracy is required
- Simpler configuration preferred

## Next Steps

- [LoRA](../LoRA/) - Standard LoRA without quantization
- [TextClassification](../../nlp/TextClassification/) - Apply fine-tuning to text tasks
- [Embeddings](../../nlp/Embeddings/) - Fine-tune embedding models
