# Model Quantization - Optimizing for Deployment

This sample demonstrates how to quantize trained models for efficient deployment using AiDotNet's quantization tools.

## Overview

Quantization reduces model size and inference latency by:
1. Converting weights from FP32 to INT8 or FP16
2. Calibrating with representative data
3. Applying post-training or quantization-aware training

## Prerequisites

- .NET 8.0 SDK or later
- AiDotNet NuGet package
- A trained model to quantize

## Running the Sample

```bash
cd samples/deployment/Quantization
dotnet run
```

## What This Sample Demonstrates

1. **Post-Training Quantization (PTQ)**: Quick quantization without retraining
2. **Quantization-Aware Training (QAT)**: Higher accuracy through training
3. **Dynamic Quantization**: Runtime quantization for flexibility
4. **Calibration**: Using representative data for accurate quantization
5. **Accuracy Evaluation**: Measuring quantization impact

## Quantization Types

### INT8 Quantization
- 4x smaller models
- Faster inference on INT8-capable hardware
- Best for deployment on edge devices

### FP16 Quantization
- 2x smaller models
- Native GPU support
- Minimal accuracy loss

### Mixed Precision
- Sensitive layers in FP32
- Other layers quantized
- Balance between accuracy and efficiency

## Results Comparison

| Quantization | Size | Accuracy | Speedup |
|--------------|------|----------|---------|
| FP32 (base) | 100% | 100% | 1.0x |
| FP16 | 50% | ~99.9% | 1.5-2x |
| INT8 (PTQ) | 25% | ~99% | 2-4x |
| INT8 (QAT) | 25% | ~99.5% | 2-4x |

## Code Structure

- `Program.cs` - Main entry point with quantization examples
- Demonstrates PTQ, QAT, and dynamic quantization
- Includes calibration and evaluation

## Related Samples

- [Model Serving](../ModelServing/) - Deploy quantized models
- [ONNX Export](../ONNX/) - Export to ONNX format
- [Pruning](../Pruning/) - Model pruning for smaller models

## Learn More

- [Quantization Guide](/docs/tutorials/quantization/)
- [Deployment Best Practices](/docs/guides/deployment-best-practices/)
- [AiDotNet.Quantization API Reference](/api/AiDotNet.Quantization/)
