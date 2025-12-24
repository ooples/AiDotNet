# AiDotNet.Serving - Usage Example

This document provides a complete, beginner-friendly example of using the AiDotNet.Serving framework.

## Complete Example: Building a House Price Predictor API

Let's build a complete example that trains a linear regression model and serves it via REST API.

### Step 1: Train Your Model

First, train a model using AiDotNet:

```csharp
using AiDotNet.Regression;
using AiDotNet.LinearAlgebra;

// Sample training data: [square_feet, bedrooms, bathrooms] -> price
var trainingData = new Matrix<double>(new[,]
{
    { 1200, 2, 1 },
    { 1500, 3, 2 },
    { 1800, 3, 2 },
    { 2000, 4, 3 },
    { 2200, 4, 3 }
});

var prices = new Vector<double>(new[]
{
    200000.0, 250000.0, 300000.0, 350000.0, 400000.0
});

// Train the model
var options = new RegressionOptions<double>
{
    UseIntercept = true,
    Regularization = RegularizationType.None
};

var model = new LinearRegression<double>(options);
model.Train(trainingData, prices);
```

### Step 2: Create a Serving Application

Create a new console application or modify your existing Program.cs:

```csharp
using AiDotNet.Serving;
using AiDotNet.Serving.Services;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Configuration;
using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;

var builder = WebApplication.CreateBuilder(args);

// Configure serving options
builder.Services.Configure<ServingOptions>(options =>
{
    options.Port = 52432;
    options.BatchingWindowMs = 10;
    options.MaxBatchSize = 100;
});

// Register services
builder.Services.AddSingleton<IModelRepository, ModelRepository>();
builder.Services.AddSingleton<IRequestBatcher, RequestBatcher>();
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Load your trained model
var repository = app.Services.GetRequiredService<IModelRepository>();

// Wrap your regression model for serving
var servableModel = new ServableModelWrapper<double>(
    modelName: "house-price-predictor",
    regressionModel: model,
    inputDimension: 3  // square_feet, bedrooms, bathrooms
);

repository.LoadModel("house-price-predictor", servableModel);

// Configure pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseAuthorization();
app.MapControllers();

Console.WriteLine("🚀 AiDotNet Serving API is running on http://localhost:52432");
Console.WriteLine("📖 Swagger UI: http://localhost:52432/swagger");
Console.WriteLine("📊 Model loaded: house-price-predictor");

app.Run();
```

### Step 3: Make Predictions

#### Using cURL

```bash
# Get list of loaded models
curl http://localhost:52432/api/models

# Make a prediction for a house with 1600 sq ft, 3 bedrooms, 2 bathrooms
curl -X POST http://localhost:52432/api/inference/predict/house-price-predictor \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1600, 3, 2]],
    "requestId": "house-1"
  }'

# Response:
# {
#   "predictions": [[267000.0]],
#   "requestId": "house-1",
#   "processingTimeMs": 12,
#   "batchSize": 1
# }
```

#### Using C# Client

```csharp
using System.Net.Http.Json;

var client = new HttpClient
{
    BaseAddress = new Uri("http://localhost:52432")
};

// Create a prediction request
var request = new PredictionRequest
{
    Features = new[]
    {
        new[] { 1600.0, 3.0, 2.0 }  // square_feet, bedrooms, bathrooms
    },
    RequestId = "house-1"
};

// Send request
var response = await client.PostAsJsonAsync(
    "/api/inference/predict/house-price-predictor",
    request
);

var result = await response.Content.ReadFromJsonAsync<PredictionResponse>();

Console.WriteLine($"Predicted price: ${result.Predictions[0][0]:N2}");
Console.WriteLine($"Processing time: {result.ProcessingTimeMs}ms");
```

#### Using Python

```python
import requests
import json

# Make a prediction
response = requests.post(
    'http://localhost:52432/api/inference/predict/house-price-predictor',
    json={
        'features': [[1600, 3, 2]],
        'requestId': 'house-1'
    }
)

result = response.json()
print(f"Predicted price: ${result['predictions'][0][0]:,.2f}")
print(f"Processing time: {result['processingTimeMs']}ms")
```

### Step 4: Test Batch Performance

Send multiple concurrent requests to see batching in action:

```csharp
// Send 50 concurrent predictions
var tasks = Enumerable.Range(0, 50).Select(i =>
{
    var sqft = 1200 + (i * 20);
    var bedrooms = 2 + (i % 3);
    var bathrooms = 1 + (i % 2);

    var request = new PredictionRequest
    {
        Features = new[] { new[] { (double)sqft, (double)bedrooms, (double)bathrooms } },
        RequestId = $"batch-request-{i}"
    };

    return client.PostAsJsonAsync(
        "/api/inference/predict/house-price-predictor",
        request
    );
}).ToArray();

var responses = await Task.WhenAll(tasks);

// Check statistics
var stats = await client.GetFromJsonAsync<Dictionary<string, object>>(
    "/api/inference/stats"
);

Console.WriteLine($"Total requests: {stats["totalRequests"]}");
Console.WriteLine($"Total batches: {stats["totalBatches"]}");
Console.WriteLine($"Average batch size: {stats["averageBatchSize"]}");

// Example output:
// Total requests: 50
// Total batches: 5
// Average batch size: 10
```

## Advanced: Multi-Output Model Example

Here's an example with a model that has multiple outputs:

```csharp
// Model that predicts price AND price_per_sqft
var multiOutputModel = new ServableModelWrapper<double>(
    modelName: "multi-output-predictor",
    inputDimension: 3,
    outputDimension: 2,
    predictFunc: input =>
    {
        var numOps = MathHelper.GetNumericOperations<double>();

        // Simple logic: price = base + sqft*100 + bedrooms*10000
        var sqft = input[0];
        var bedrooms = input[1];
        var bathrooms = input[2];

        var price = numOps.Add(
            numOps.FromDouble(100000),
            numOps.Add(
                numOps.Multiply(sqft, numOps.FromDouble(100)),
                numOps.Multiply(bedrooms, numOps.FromDouble(10000))
            )
        );

        var pricePerSqft = numOps.Divide(price, sqft);

        return new Vector<double>(new[] { price, pricePerSqft });
    },
    predictBatchFunc: inputs =>
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var result = new Matrix<double>(inputs.Rows, 2);

        for (int i = 0; i < inputs.Rows; i++)
        {
            var sqft = inputs[i, 0];
            var bedrooms = inputs[i, 1];

            var price = numOps.Add(
                numOps.FromDouble(100000),
                numOps.Add(
                    numOps.Multiply(sqft, numOps.FromDouble(100)),
                    numOps.Multiply(bedrooms, numOps.FromDouble(10000))
                )
            );

            var pricePerSqft = numOps.Divide(price, sqft);

            result[i, 0] = price;
            result[i, 1] = pricePerSqft;
        }

        return result;
    }
);

repository.LoadModel("multi-output-predictor", multiOutputModel);

// Use it:
var response = await client.PostAsJsonAsync(
    "/api/inference/predict/multi-output-predictor",
    new PredictionRequest
    {
        Features = new[] { new[] { 1600.0, 3.0, 2.0 } }
    }
);

var result = await response.Content.ReadFromJsonAsync<PredictionResponse>();
Console.WriteLine($"Price: ${result.Predictions[0][0]:N2}");
Console.WriteLine($"Price per sqft: ${result.Predictions[0][1]:N2}");
```

## Performance Tuning

### Optimizing Batch Window

```csharp
builder.Services.Configure<ServingOptions>(options =>
{
    // For real-time applications (low latency priority)
    options.BatchingWindowMs = 5;

    // For batch processing (high throughput priority)
    // options.BatchingWindowMs = 50;

    options.MaxBatchSize = 100;
});
```

### Monitoring Performance

```csharp
// Periodically check statistics
var timer = new Timer(async _ =>
{
    var stats = await client.GetFromJsonAsync<Dictionary<string, object>>(
        "/api/inference/stats"
    );

    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] " +
        $"Requests: {stats["totalRequests"]}, " +
        $"Batches: {stats["totalBatches"]}, " +
        $"Avg batch size: {stats["averageBatchSize"]}");

}, null, TimeSpan.Zero, TimeSpan.FromSeconds(10));
```

## Production Checklist

Before deploying to production:

- [ ] Implement proper model serialization/deserialization
- [ ] Add authentication and authorization to endpoints
- [ ] Set up proper logging and monitoring
- [ ] Configure HTTPS
- [ ] Set appropriate `MaxBatchSize` based on memory constraints
- [ ] Tune `BatchingWindowMs` based on your latency requirements
- [ ] Add health check endpoints
- [ ] Implement graceful shutdown for model cleanup
- [ ] Set up load testing to verify performance
- [ ] Configure CORS policies appropriately

## Troubleshooting

### High Latency

```csharp
// Reduce batching window
options.BatchingWindowMs = 5;  // Faster response, smaller batches
```

### Memory Issues

```csharp
// Limit batch size
options.MaxBatchSize = 50;  // Process smaller batches
```

### Not Batching

```csharp
// Increase batching window to collect more requests
options.BatchingWindowMs = 20;  // Wait longer to collect requests

// Verify concurrent load
// Send multiple requests within the batching window
var tasks = Enumerable.Range(0, 10)
    .Select(i => SendPredictionRequestAsync())
    .ToArray();
await Task.WhenAll(tasks);
```

## Next Steps

1. Review the [README.md](README.md) for complete API documentation
2. Check out the [integration tests](../../tests/AiDotNet.Serving.Tests/ServingIntegrationTests.cs) for more examples
3. Explore the [Swagger UI](http://localhost:52432/swagger) for interactive API testing

---

**Questions?** Open an issue on the [AiDotNet repository](https://github.com/ooples/AiDotNet/issues).
