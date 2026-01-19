# LoRA Fine-Tuning - Parameter-Efficient Adaptation

This sample demonstrates Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning using AiDotNet, showing how to adapt pre-trained models with 10-100x fewer trainable parameters.

## What You'll Learn

- How LoRA reduces trainable parameters through low-rank decomposition
- How to configure and apply LoRA adapters to neural network layers
- How to track training loss and monitor gradient norms
- How to merge LoRA adapters back into the base model
- How different LoRA ranks affect compression and capacity

## What is LoRA?

LoRA (Low-Rank Adaptation) freezes pre-trained model weights and injects trainable low-rank decomposition matrices:

```
Standard Fine-Tuning:
  W_new = W_original + Delta_W      (Delta_W has millions of parameters)

LoRA Fine-Tuning:
  W_new = W_original + B * A        (B and A have thousands of parameters)

Where:
  W_original: [d_out x d_in] frozen pre-trained weights
  A: [rank x d_in] trainable down-projection
  B: [d_out x rank] trainable up-projection
  rank << min(d_out, d_in)
```

## Running the Sample

```bash
cd samples/llm-fine-tuning/LoRA
dotnet run
```

## Expected Output

```
=== AiDotNet LoRA Fine-Tuning ===
Parameter-Efficient Fine-Tuning with Low-Rank Adaptation

Configuration:
  - Input Size: 128
  - Hidden Size: 256
  - Output Size: 10
  - LoRA Rank: 8
  - LoRA Alpha: 8.0
  - Learning Rate: 0.001

Creating base model (simulating pre-trained weights)...
  Base model parameters: 101,386

Applying LoRA adapters...
  - Rank: 8
  - Alpha: 8.0
  - Freeze Base Layers: True

LoRA Adapter Configuration:
  Layer 1: 3,072 trainable parameters
  Layer 2: 4,096 trainable parameters
  Layer 3: 2,128 trainable parameters
  Total LoRA parameters: 9,296
  Parameter reduction: 90.8%
  Compression ratio: 10.9x

Generating training data...
  Training samples: 1000
  Validation samples: 200

============================================================
Training with LoRA (Standard)
============================================================

Epoch  | Train Loss | Val Loss   | LoRA Grad Norm | Time
------------------------------------------------------------
    1  |     2.3026 |     2.3021 |       0.012345 |   45ms
   11  |     1.8234 |     1.8456 |       0.008234 |   42ms
   21  |     1.4567 |     1.4789 |       0.005678 |   41ms
   ...
  100  |     0.4123 |     0.4567 |       0.001234 |   40ms
------------------------------------------------------------
Total training time: 4.21s
Average epoch time: 42.1ms

============================================================
Training Summary
============================================================

Initial Loss: 2.3026
Final Loss: 0.4123
Loss Reduction: 82.1%

Best Validation Loss: 0.4234 (Epoch 95)

Training Loss Curve:
  2.303 |
         |*
         | ***
         |    ****
         |        *****
         |             *******
         |                    **********
  0.412 |--------------------------------------------------
         0                    Epoch                     100

============================================================
Final Model Evaluation
============================================================

Validation Accuracy: 87.50%

============================================================
LoRA Adapter Merging
============================================================

Merging LoRA adapters into base layers...
  Layer 1 merged successfully
  Layer 2 merged successfully
  Layer 3 merged successfully

Merging complete! Merged model has:
  - Same parameter count as base model: 101,386
  - LoRA adaptations baked into weights
  - No additional inference overhead

============================================================
LoRA Rank Comparison
============================================================

| Rank | Trainable Params | Compression | Memory Savings |
|------|------------------|-------------|----------------|
|    1 |            1,162 |       87.3x |          98.9% |
|    4 |            4,648 |       21.8x |          95.4% |
|    8 |            9,296 |       10.9x |          90.8% |
|   16 |           18,592 |        5.5x |          81.7% |
|   32 |           37,184 |        2.7x |          63.3% |
|   64 |           74,368 |        1.4x |          26.7% |
```

## How LoRA Works

### 1. Low-Rank Decomposition

Instead of updating the full weight matrix, LoRA adds a low-rank update:

```
Forward pass:
  y = W_0 * x + (B * A) * x * (alpha / rank)

Where:
  W_0: Original frozen weights
  B: [d_out x rank] matrix (initialized to zero)
  A: [rank x d_in] matrix (initialized with Kaiming)
  alpha: Scaling factor
  rank: Rank of decomposition (e.g., 8)
```

### 2. Parameter Savings

For a weight matrix of size [1024 x 1024]:

| Method | Parameters | Relative |
|--------|------------|----------|
| Full Fine-Tuning | 1,048,576 | 100% |
| LoRA (rank=64) | 131,072 | 12.5% |
| LoRA (rank=16) | 32,768 | 3.1% |
| LoRA (rank=8) | 16,384 | 1.6% |
| LoRA (rank=4) | 8,192 | 0.8% |

### 3. Merging for Inference

After training, LoRA weights can be merged into the base model:

```csharp
// Training: separate adapters
y = W_0 * x + (B * A) * x

// After merging: single weight matrix
W_merged = W_0 + B * A
y = W_merged * x  // Same result, no overhead!
```

## Code Highlights

### Creating LoRA Configuration

```csharp
var loraConfig = new DefaultLoRAConfiguration<double>(
    rank: 8,           // Low-rank dimension
    alpha: 8.0,        // Scaling factor
    freezeBaseLayer: true  // Freeze pre-trained weights
);
```

### Applying LoRA to Layers

```csharp
// Wrap existing layer with LoRA adapter
var loraLayer = loraConfig.ApplyLoRA(baseLayer);

// DefaultLoRAConfiguration automatically:
// - Wraps Dense layers with StandardLoRAAdapter
// - Wraps Convolutional layers with StandardLoRAAdapter
// - Leaves activation/pooling layers unchanged
```

### Training Loop

```csharp
// Forward pass through LoRA layers
var output1 = loraLayer1.Forward(input);
var output2 = loraLayer2.Forward(output1);
var output3 = loraLayer3.Forward(output2);

// Backward pass (only LoRA parameters receive gradients)
var grad3 = loraLayer3.Backward(lossGradient);
var grad2 = loraLayer2.Backward(grad3);
var grad1 = loraLayer1.Backward(grad2);

// Update only LoRA parameters
optimizer.Step();
```

### Merging Adapters

```csharp
// After training, merge for efficient inference
if (loraLayer is ILoRAAdapter<double> adapter)
{
    var mergedLayer = adapter.MergeToOriginalLayer();
    // mergedLayer is a standard DenseLayer with LoRA baked in
}
```

## LoRA Variants in AiDotNet

AiDotNet includes 32 LoRA variants:

| Variant | Use Case | Benefit |
|---------|----------|---------|
| `StandardLoRAAdapter` | General purpose | Simple, effective |
| `QLoRAAdapter` | Memory-constrained | 4-bit quantization |
| `DoRAAdapter` | Better accuracy | Weight decomposition |
| `AdaLoRAAdapter` | Dynamic rank | Adaptive allocation |
| `VeRAAdapter` | Extreme efficiency | 10x fewer parameters |
| `LoRAPlusAdapter` | Faster convergence | Dual learning rates |

## Architecture

```
                Input
                  |
                  v
    +---------------------------+
    |     Base Layer (Frozen)   |
    |        W_0 * x            |
    +------------+--------------+
                 |
    +------------+--------------+
    |      LoRA Adapter         |
    |   B * A * x * (a/r)       |
    +------------+--------------+
                 |
                 v
    +---------------------------+
    |          Sum              |
    |   W_0*x + B*A*x*(a/r)     |
    +------------+--------------+
                 |
                 v
              Output
```

## Best Practices

1. **Choosing Rank**:
   - Start with rank=8 (good default)
   - Increase if model underfits
   - Decrease for more efficiency

2. **Alpha Selection**:
   - Common: alpha = rank
   - Higher alpha = stronger adaptation
   - Lower alpha = more conservative

3. **Which Layers**:
   - Apply to attention Q, K, V projections
   - Apply to feed-forward layers
   - Skip normalization layers

4. **Learning Rate**:
   - LoRA often needs higher LR than full fine-tuning
   - Try 1e-4 to 1e-3

## Next Steps

- [QLoRA](../QLoRA/) - 4-bit quantized LoRA for even more efficiency
- [TextClassification](../../nlp/TextClassification/) - Apply LoRA to text classifiers
- [Embeddings](../../nlp/Embeddings/) - Fine-tune embedding models with LoRA
