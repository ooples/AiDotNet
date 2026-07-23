# Deployment Samples

This directory contains examples for deploying AiDotNet models to production.

## Available Samples

| Sample | Description |
|--------|-------------|
| [ModelServing](./ModelServing/) | REST API serving with AiDotNet.Serving |
| [Quantization](./Quantization/) | Model optimization for deployment |

## Deployment Options

### 1. AiDotNet.Serving (Recommended)
Production-ready model serving with:
- Dynamic request batching
- Model hot-loading
- API key authentication
- Tier-based access control
- Swagger documentation

```csharp
// See ModelServing sample for complete example
var repository = app.Services.GetRequiredService<IModelRepository>();
repository.LoadModel("my-model", servableModel, modelPath);
```

### 2. Model Quantization
Optimize models before deployment:
- INT8 quantization (4x smaller)
- FP16 quantization (2x smaller)
- Dynamic quantization

### 3. ONNX Export
Export to ONNX for cross-platform deployment:
```csharp
model.ExportToONNX("model.onnx");
```

## Model Compression (10+)

| Method | Size Reduction | Accuracy Impact |
|--------|---------------|-----------------|
| FP16 | 2x | Minimal |
| INT8 (PTQ) | 4x | ~1% loss |
| INT8 (QAT) | 4x | ~0.5% loss |
| Pruning | 2-10x | Varies |
| Distillation | N/A | Varies |

## Learn More

- [Deployment Guide](/docs/tutorials/deployment/)
- [Quantization Guide](/docs/guides/quantization/)
- [AiDotNet.Serving Documentation](/api/AiDotNet.Serving/)
