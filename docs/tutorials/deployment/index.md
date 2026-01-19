---
layout: default
title: Deployment
parent: Tutorials
nav_order: 10
has_children: true
permalink: /tutorials/deployment/
---

# Deployment Tutorial
{: .no_toc }

Deploy your trained models to production with AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AiDotNet provides multiple deployment options:
- **AiDotNet.Serving**: Production REST API server
- **Model Quantization**: Optimize for inference
- **ONNX Export**: Cross-platform deployment
- **Edge Deployment**: Mobile and IoT

---

## AiDotNet.Serving

### Quick Start

```csharp
using AiDotNet.Serving;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Services;

var builder = WebApplication.CreateBuilder(args);

// Configure serving options
builder.Services.Configure<ServingOptions>(options =>
{
    options.Port = 8080;
    options.BatchingWindowMs = 10;
    options.MaxBatchSize = 32;
});

// Register services
builder.Services.AddSingleton<IModelRepository, ModelRepository>();
builder.Services.AddSingleton<IRequestBatcher, RequestBatcher>();
builder.Services.AddControllers();

var app = builder.Build();

// Load model
var repository = app.Services.GetRequiredService<IModelRepository>();
repository.LoadModel("my-model", servableModel, modelPath);

app.MapControllers();
app.Run();
```

### API Endpoints

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/api/models` | GET | List loaded models |
| `/api/models` | POST | Load a model |
| `/api/models/{name}` | DELETE | Unload a model |
| `/api/inference/predict/{name}` | POST | Make prediction |
| `/api/inference/stats` | GET | Batching statistics |

### Making Predictions

```bash
curl -X POST http://localhost:8080/api/inference/predict/my-model \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0, 4.0]]}'
```

---

## Model Quantization

### Post-Training Quantization (PTQ)

```csharp
using AiDotNet.Quantization;

var config = new QuantizationConfig<float>
{
    QuantizationType = QuantizationType.INT8,
    CalibrationMethod = CalibrationMethod.MinMax,
    CalibrationSamples = 100
};

var quantizer = new ModelQuantizer<float>(config);
quantizer.Calibrate(model, calibrationData);
var quantizedModel = quantizer.Quantize(model);
```

### Quantization-Aware Training (QAT)

```csharp
var qatConfig = new QATConfig<float>
{
    TargetQuantization = QuantizationType.INT8,
    SimulateQuantization = true,
    QATEpochs = 10
};

var qatWrapper = new QATWrapper<float>(model, qatConfig);
qatWrapper.EnableFakeQuantization();

for (int epoch = 0; epoch < qatConfig.QATEpochs; epoch++)
{
    qatWrapper.Train(trainData, trainLabels);
}

var quantizedModel = qatWrapper.ConvertToQuantized();
```

### Results Comparison

| Method | Size | Accuracy | Speedup |
|:-------|:-----|:---------|:--------|
| FP32 | 100% | Baseline | 1.0x |
| FP16 | 50% | ~99.9% | 1.5x |
| INT8 (PTQ) | 25% | ~99% | 2-4x |
| INT8 (QAT) | 25% | ~99.5% | 2-4x |

---

## Model Saving and Loading

### Save Trained Model

```csharp
// After training
var result = await builder.BuildAsync(trainData, trainLabels);

// Save to file
result.SaveToFile("model.aidotnet");
```

### Load for Inference

```csharp
// Load model
var loadedResult = AiModelResult<double, double[], double>
    .LoadFromFile("model.aidotnet");

// Make predictions
var prediction = loadedResult.Model.Predict(input);
```

---

## ONNX Export

Export to ONNX for cross-platform deployment:

```csharp
using AiDotNet.ONNX;

// Export to ONNX
model.ExportToONNX("model.onnx", new ONNXExportConfig
{
    OpsetVersion = 17,
    InputNames = ["input"],
    OutputNames = ["output"],
    DynamicAxes = new Dictionary<string, int[]>
    {
        ["input"] = [0],   // Batch dimension
        ["output"] = [0]
    }
});
```

### Use with ONNX Runtime

```csharp
using Microsoft.ML.OnnxRuntime;

using var session = new InferenceSession("model.onnx");

var input = new DenseTensor<float>(data, shape);
var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", input) };

using var results = session.Run(inputs);
var output = results.First().AsEnumerable<float>().ToArray();
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 8080

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY . .
RUN dotnet publish -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=build /app/publish .
COPY models/ /app/models/
ENTRYPOINT ["dotnet", "ModelServer.dll"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  model-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    environment:
      - ASPNETCORE_ENVIRONMENT=Production
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: myregistry/model-server:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: model-server
spec:
  selector:
    app: model-server
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

---

## Performance Optimization

### Batching

```csharp
builder.Services.Configure<ServingOptions>(options =>
{
    options.BatchingWindowMs = 10;  // Wait 10ms to collect requests
    options.MaxBatchSize = 64;      // Max batch size
});
```

### Caching

```csharp
// Enable response caching for repeated inputs
builder.Services.AddResponseCaching();
app.UseResponseCaching();
```

### Connection Pooling

```csharp
// For database-backed model stores
builder.Services.AddDbContextPool<ModelDbContext>(options =>
    options.UseSqlServer(connectionString));
```

---

## Monitoring

### Health Checks

```csharp
builder.Services.AddHealthChecks()
    .AddCheck<ModelHealthCheck>("models")
    .AddCheck<GpuHealthCheck>("gpu");

app.MapHealthChecks("/health");
app.MapHealthChecks("/ready", new HealthCheckOptions
{
    Predicate = check => check.Tags.Contains("ready")
});
```

### Metrics

```csharp
// Prometheus metrics endpoint
app.MapGet("/metrics", (IMetricsCollector metrics) =>
{
    return Results.Text(metrics.GetPrometheusMetrics(), "text/plain");
});
```

---

## Best Practices

1. **Quantize for production**: INT8 gives 2-4x speedup
2. **Enable batching**: Improves throughput significantly
3. **Use health checks**: For load balancer integration
4. **Monitor latency**: Track p50, p95, p99
5. **Horizontal scaling**: Use Kubernetes for auto-scaling
6. **GPU sharing**: Multiple models on one GPU with CUDA MPS

---

## Next Steps

- [ModelServing Sample](/samples/deployment/ModelServing/)
- [Quantization Sample](/samples/deployment/Quantization/)
- [AiDotNet.Serving API Reference](/api/AiDotNet.Serving/)
