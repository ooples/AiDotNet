# AiDotNet.Serving

A production-ready REST API server for deploying trained AiDotNet models with **dynamic request batching** to maximize throughput.

## 🚀 Overview

AiDotNet.Serving provides a high-performance model serving framework that allows you to deploy your trained machine learning models as REST APIs. The key feature is **dynamic request batching** - multiple concurrent inference requests are automatically collected and processed as a single batch, significantly improving throughput.

## ✨ Key Features

- **🔄 Dynamic Request Batching**: Automatically batches concurrent requests for optimal performance
- **🔒 Thread-Safe Model Management**: Load, list, and unload models safely from multiple threads
- **📊 Multiple Numeric Types**: Support for `double`, `float`, and `decimal` numeric types
- **📈 Performance Statistics**: Real-time metrics on batching performance
- **📝 OpenAPI/Swagger Documentation**: Interactive API documentation out of the box
- **🧪 Comprehensive Tests**: Full test coverage including batch processing verification

## 🏗️ Architecture

The framework consists of three main components:

### 1. ModelRepository (Singleton)
Thread-safe storage for loaded models using `ConcurrentDictionary`.

```csharp
public interface IModelRepository
{
    bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null);
    IServableModel<T>? GetModel<T>(string name);
    bool UnloadModel(string name);
    List<ModelInfo> GetAllModelInfo();
}
```

### 2. RequestBatcher (Singleton)
Collects incoming requests and processes them in batches:
- Requests are queued in a bounded `Channel` for backpressure
- A background processing loop triggers batch formation and execution
- Requests are grouped by model and numeric type
- The model's `PredictBatch` method is called once for the entire batch
- Individual results are returned via `TaskCompletionSource`

```csharp
public interface IRequestBatcher
{
    Task<Vector<T>> QueueRequest<T>(string modelName, Vector<T> input);
    Dictionary<string, object> GetStatistics();
}
```

### 3. REST API Controllers

#### ModelsController
- `POST /api/models` - Load a model artifact from a file under the configured model directory
- `GET /api/models` - List all loaded models
- `GET /api/models/{name}` - Get specific model info
- `DELETE /api/models/{name}` - Unload a model

#### InferenceController
- `POST /api/inference/predict/{modelName}` - Perform prediction (automatically batched)
- `GET /api/inference/stats` - Get batching statistics

## 📦 Installation

### Prerequisites
- .NET 8.0 SDK or later
- AiDotNet library

### Adding to Your Solution

```bash
# Add the serving project reference
dotnet add reference src/AiDotNet.Serving/AiDotNet.Serving.csproj

# Add the test project reference (optional)
dotnet add reference tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj
```

## 🎯 Quick Start

### 1. Basic Usage

```csharp
using AiDotNet.Serving;
using AiDotNet.Serving.Services;
using AiDotNet.Serving.Models;
using AiDotNet.LinearAlgebra;

// Start the server (in Program.cs, this is done automatically)
var builder = WebApplication.CreateBuilder(args);
// ... configuration (see Program.cs)
var app = builder.Build();
app.Run();
```

### 2. Loading a Model Programmatically

Since file-based model serialization is application-specific, you'll typically load models programmatically:

```csharp
// Get the model repository from dependency injection
var repository = app.Services.GetRequiredService<IModelRepository>();

// Option 1: Wrap an existing regression model
var regressionModel = new LinearRegression<double>(options);
regressionModel.Train(X, y);

var servableModel = new ServableModelWrapper<double>(
    modelName: "my-linear-model",
    regressionModel: regressionModel,
    inputDimension: 10  // Number of features
);

repository.LoadModel("my-linear-model", servableModel);

// Option 2: Create a custom model with prediction functions
var customModel = new ServableModelWrapper<double>(
    modelName: "custom-model",
    inputDimension: 5,
    outputDimension: 3,
    predictFunc: input => {
        // Your prediction logic here
        return new Vector<double>(new[] { 1.0, 2.0, 3.0 });
    },
    predictBatchFunc: inputs => {
        // Optimized batch prediction
        // Process all inputs at once
        return batchResults;
    }
);

repository.LoadModel("custom-model", customModel);
```

### 3. Making Predictions

Once the server is running, you can make predictions via HTTP:

```bash
# List loaded models
curl http://localhost:5000/api/models

# Make a prediction
curl -X POST http://localhost:5000/api/inference/predict/my-linear-model \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
    "requestId": "test-request-1"
  }'

# Get batching statistics
curl http://localhost:5000/api/inference/stats

# Unload a model
curl -X DELETE http://localhost:5000/api/models/my-linear-model
```

### 4. Using the Swagger UI

Navigate to `http://localhost:5000` to access the interactive Swagger UI where you can:
- View all available endpoints
- Test API calls directly in the browser
- See request/response schemas
- Read detailed endpoint documentation

## ⚙️ Configuration

Configure the serving framework in `appsettings.json`:

```json
{
  "ServingOptions": {
    "Port": 5000,
    "BatchingWindowMs": 10,
    "MaxBatchSize": 100,
    "StartupModels": []
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `Port` | int | 5000 | Port number for the HTTP server |
| `BatchingWindowMs` | int | 10 | Time window in milliseconds to collect requests before processing |
| `MaxBatchSize` | int | 100 | Maximum number of requests to batch together (0 = unlimited) |
| `StartupModels` | array | [] | Models to load at startup (requires custom implementation) |

### Tuning Batch Performance

The `BatchingWindowMs` parameter is critical for performance:
- **Lower values (5-10ms)**: Lower latency per request, smaller batches
- **Higher values (20-50ms)**: Higher throughput, larger batches, but higher latency

Choose based on your use case:
- **Real-time applications**: 5-10ms for low latency
- **Batch processing**: 20-50ms for maximum throughput

## 🧪 Testing

The framework includes comprehensive integration tests:

```bash
# Run all tests
dotnet test tests/AiDotNet.Serving.Tests/

# Run with coverage
dotnet test /p:CollectCoverage=true /p:CoverletOutputFormat=opencover
```

### Key Test Coverage

- ✅ Model management (load, list, unload)
- ✅ Basic inference operations
- ✅ **Batch processing verification** (critical test proving models are called once per batch)
- ✅ Error handling (404, 400 responses)
- ✅ Concurrent request handling
- ✅ Statistics tracking

## 📊 Monitoring

### Batch Statistics

Monitor the performance of the request batcher:

```bash
curl http://localhost:5000/api/inference/stats
```

Response:
```json
{
  "totalRequests": 1000,
  "totalBatches": 25,
  "queuedRequests": 0,
  "averageBatchSize": 40.0
}
```

### Logging

The framework uses `ILogger` for diagnostic logging. Set log levels in `appsettings.json`:

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "AiDotNet.Serving": "Debug"
    }
  }
}
```

## 🎓 Examples

### Example 1: Simple Linear Model

```csharp
// Train a model
var model = new LinearRegression<double>();
model.Train(trainingX, trainingY);

// Wrap it for serving
var servable = new ServableModelWrapper<double>(
    "price-predictor",
    model,
    inputDimension: trainingX.Columns
);

// Load it into the repository
repository.LoadModel("price-predictor", servable);
```

### Example 2: Custom Neural Network

```csharp
// Assuming you have a neural network model
var neuralNet = new MyNeuralNetwork<double>();
neuralNet.Train(X, y);

var servable = new ServableModelWrapper<double>(
    modelName: "my-neural-net",
    inputDimension: 100,
    outputDimension: 10,
    predictFunc: input => neuralNet.Forward(input),
    predictBatchFunc: inputs => neuralNet.ForwardBatch(inputs)
);

repository.LoadModel("my-neural-net", servable);
```

### Example 3: Concurrent Predictions (Client Side)

```csharp
using System.Net.Http.Json;

var client = new HttpClient { BaseAddress = new Uri("http://localhost:5000") };

// Send 100 concurrent requests - they'll be automatically batched
var tasks = Enumerable.Range(0, 100).Select(async i =>
{
    var request = new PredictionRequest
    {
        Features = new[] { GenerateFeatures(i) }
    };

    var response = await client.PostAsJsonAsync(
        "/api/inference/predict/my-model",
        request
    );

    return await response.Content.ReadFromJsonAsync<PredictionResponse>();
});

var results = await Task.WhenAll(tasks);

// Check statistics - you should see averageBatchSize > 1
var stats = await client.GetFromJsonAsync<Dictionary<string, object>>(
    "/api/inference/stats"
);
Console.WriteLine($"Average batch size: {stats["averageBatchSize"]}");
```

## 🔧 Advanced Usage

### Custom Model Interface

To create a fully custom model without using the wrapper:

```csharp
public class MyCustomModel : IServableModel<double>
{
    public string ModelName => "my-custom-model";
    public int InputDimension => 10;
    public int OutputDimension => 5;

    public Vector<double> Predict(Vector<double> input)
    {
        // Your prediction logic
        return new Vector<double>(OutputDimension);
    }

    public Matrix<double> PredictBatch(Matrix<double> inputs)
    {
        // Optimized batch prediction
        return new Matrix<double>(inputs.Rows, OutputDimension);
    }
}
```

### Model Serialization

AiDotNet.Serving loads and serves AiDotNet model artifacts from disk via `PredictionModelResult<T, Matrix<T>, Vector<T>>`.
The server validates that the requested model path resolves under `ServingOptions.ModelDirectory` to prevent directory traversal.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. All existing tests pass
2. New features include tests
3. Code coverage remains above 80%
4. XML documentation is provided for public APIs

## 📝 License

This project is part of the AiDotNet library. See the main repository for license information.

## 🐛 Troubleshooting

### Models not batching

- Check `BatchingWindowMs` is not too low (< 5ms)
- Ensure you're sending concurrent requests
- Check statistics: `curl http://localhost:5000/api/inference/stats`

### High latency

- Reduce `BatchingWindowMs` (try 5ms)
- Check if `MaxBatchSize` is causing delays

### Model not found errors

- Verify the model is loaded: `curl http://localhost:5000/api/models`
- Check the model name matches exactly (case-sensitive)
- Ensure the numeric type matches (double/float/decimal)

## 📚 Additional Resources

- [AiDotNet Main Repository](https://github.com/ooples/AiDotNet)
- [ASP.NET Core Documentation](https://docs.microsoft.com/aspnet/core)
- [xUnit Testing Documentation](https://xunit.net/)

---

**Note**: This serving framework is designed for production use but requires proper model serialization implementation for file-based loading. The current implementation focuses on programmatic model loading, which gives you full flexibility in how models are persisted and loaded.
