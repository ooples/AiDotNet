# Model Serving with AiDotNet.Serving

This sample demonstrates how to deploy trained AiDotNet models in production using the **existing AiDotNet.Serving library**.

## Overview

AiDotNet.Serving is a production-ready model serving framework that provides:
1. High-performance REST API endpoints
2. Dynamic request batching for throughput optimization
3. Model hot-loading and unloading
4. Tier-based access control and API key authentication
5. Swagger/OpenAPI documentation
6. Federated learning support
7. Model artifact encryption

## Prerequisites

- .NET 8.0 SDK or later
- AiDotNet NuGet package
- AiDotNet.Serving project reference

## Running the Sample

```bash
cd samples/deployment/ModelServing
dotnet run
```

The server will start on `http://localhost:5100`.

## What This Sample Demonstrates

1. **Model Loading**: Loading saved models for inference
2. **REST API**: Exposing prediction endpoints
3. **Batching**: Efficient batch processing of requests
4. **Health Checks**: Readiness and liveness probes
5. **Metrics**: Prometheus-compatible metrics endpoint

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |
| `/models` | GET | List loaded models |

## Request Format

### Single Prediction
```json
POST /predict
{
    "model": "image-classifier",
    "input": [0.5, 0.3, 0.8, ...]
}
```

### Batch Prediction
```json
POST /predict/batch
{
    "model": "image-classifier",
    "inputs": [
        [0.5, 0.3, 0.8, ...],
        [0.2, 0.7, 0.1, ...]
    ]
}
```

## Configuration

Configure via `appsettings.json`:

```json
{
    "ModelServing": {
        "Models": [
            {
                "Name": "image-classifier",
                "Path": "./models/classifier.aidotnet",
                "Replicas": 2
            }
        ],
        "Batching": {
            "MaxBatchSize": 32,
            "MaxWaitMs": 50
        }
    }
}
```

## Docker Deployment

```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0
COPY ./publish /app
WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["dotnet", "ModelServing.dll"]
```

## Code Structure

- `Program.cs` - Server configuration and startup
- `Controllers/` - API controllers
- `Services/` - Model management services
- `Models/` - Request/response DTOs

## Related Samples

- [Quantization](../Quantization/) - Model optimization for deployment
- [ONNX Export](../ONNX/) - Export to ONNX format
- [Kubernetes Deployment](../Kubernetes/) - K8s deployment guide

## Learn More

- [Deployment Guide](/docs/tutorials/deployment/)
- [Performance Tuning](/docs/guides/performance-tuning/)
- [AiDotNet.Serving API Reference](/api/AiDotNet.Serving/)
