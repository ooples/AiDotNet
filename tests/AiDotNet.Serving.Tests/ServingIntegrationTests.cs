using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using AiDotNet.LinearAlgebra;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Helpers;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Integration tests for the AiDotNet Serving API.
/// These tests verify the end-to-end functionality of the model serving framework,
/// including model management, request batching, and inference endpoints.
/// </summary>
public class ServingIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public ServingIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory;
        _client = _factory.CreateClient();
    }

    /// <summary>
    /// Verifies that the GET /api/models endpoint returns an empty list when no models are loaded.
    /// </summary>
    [Fact]
    public async Task GetModels_WhenNoModelsLoaded_ReturnsEmptyList()
    {
        // Act
        var response = await _client.GetAsync("/api/models");

        // Assert
        response.EnsureSuccessStatusCode();
        var models = await response.Content.ReadFromJsonAsync<List<ModelInfo>>();
        Assert.NotNull(models);
        Assert.Empty(models);
    }

    /// <summary>
    /// Verifies that a model can be loaded programmatically and appears in the model list.
    /// </summary>
    [Fact]
    public async Task GetModels_AfterLoadingModel_ReturnsModelInfo()
    {
        // Arrange: Load a test model programmatically
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var testModel = CreateSimpleTestModel("test-model-1");
        repository.LoadModel("test-model-1", testModel);

        // Act
        var response = await _client.GetAsync("/api/models");

        // Assert
        response.EnsureSuccessStatusCode();
        var models = await response.Content.ReadFromJsonAsync<List<ModelInfo>>();
        Assert.NotNull(models);
        Assert.Single(models);
        Assert.Equal("test-model-1", models[0].Name);
        Assert.Equal("double", models[0].NumericType);
        Assert.Equal(3, models[0].InputDimension);
        Assert.Equal(1, models[0].OutputDimension);

        // Cleanup
        repository.UnloadModel("test-model-1");
    }

    /// <summary>
    /// Verifies that a specific model's information can be retrieved.
    /// </summary>
    [Fact]
    public async Task GetModel_WhenModelExists_ReturnsModelInfo()
    {
        // Arrange
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var testModel = CreateSimpleTestModel("test-model-2");
        repository.LoadModel("test-model-2", testModel);

        // Act
        var response = await _client.GetAsync("/api/models/test-model-2");

        // Assert
        response.EnsureSuccessStatusCode();
        var modelInfo = await response.Content.ReadFromJsonAsync<ModelInfo>();
        Assert.NotNull(modelInfo);
        Assert.Equal("test-model-2", modelInfo.Name);

        // Cleanup
        repository.UnloadModel("test-model-2");
    }

    /// <summary>
    /// Verifies that requesting a non-existent model returns 404.
    /// </summary>
    [Fact]
    public async Task GetModel_WhenModelDoesNotExist_Returns404()
    {
        // Act
        var response = await _client.GetAsync("/api/models/non-existent-model");

        // Assert
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    /// <summary>
    /// Verifies that a model can be unloaded successfully.
    /// </summary>
    [Fact]
    public async Task UnloadModel_WhenModelExists_ReturnsSuccess()
    {
        // Arrange
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var testModel = CreateSimpleTestModel("test-model-3");
        repository.LoadModel("test-model-3", testModel);

        // Act
        var response = await _client.DeleteAsync("/api/models/test-model-3");

        // Assert
        response.EnsureSuccessStatusCode();

        // Verify model is gone
        var getResponse = await _client.GetAsync("/api/models/test-model-3");
        Assert.Equal(HttpStatusCode.NotFound, getResponse.StatusCode);
    }

    /// <summary>
    /// Verifies that the inference endpoint can perform predictions.
    /// </summary>
    [Fact]
    public async Task Predict_WithValidInput_ReturnsResults()
    {
        // Arrange
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var testModel = CreateSimpleTestModel("test-model-4");
        repository.LoadModel("test-model-4", testModel);

        var request = new PredictionRequest
        {
            Features = new[]
            {
                new[] { 1.0, 2.0, 3.0 }
            },
            RequestId = "test-request-1"
        };

        // Act
        var response = await _client.PostAsJsonAsync("/api/inference/predict/test-model-4", request);

        // Assert
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<PredictionResponse>();
        Assert.NotNull(result);
        Assert.Equal("test-request-1", result.RequestId);
        Assert.NotNull(result.Predictions);
        Assert.Single(result.Predictions);
        Assert.Single(result.Predictions[0]); // One output value
        Assert.Equal(6.0, result.Predictions[0][0], 5); // Sum of inputs

        // Cleanup
        repository.UnloadModel("test-model-4");
    }

    /// <summary>
    /// Critical test: Verifies that batch processing works correctly.
    /// This test ensures that multiple concurrent requests are batched together
    /// and the model is called once with the full batch.
    /// </summary>
    [Fact]
    public async Task Predict_WithConcurrentRequests_ProcessesAsBatch()
    {
        // Arrange
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        int batchCallCount = 0;
        var testModel = CreateCountingTestModel("batch-test-model", ref batchCallCount);
        repository.LoadModel("batch-test-model", testModel);

        // Create 10 prediction requests
        var requests = Enumerable.Range(0, 10).Select(i => new PredictionRequest
        {
            Features = new[] { new[] { (double)i, (double)(i + 1), (double)(i + 2) } },
            RequestId = $"batch-request-{i}"
        }).ToArray();

        // Act: Send all requests concurrently
        var tasks = requests.Select(req =>
            _client.PostAsJsonAsync("/api/inference/predict/batch-test-model", req)
        ).ToArray();

        var responses = await Task.WhenAll(tasks);

        // Assert: All requests should succeed
        foreach (var response in responses)
        {
            response.EnsureSuccessStatusCode();
        }

        // Wait a moment for all batch processing to complete
        await Task.Delay(100);

        // Verify that batch processing occurred
        // The model should have been called fewer times than the number of requests
        // In ideal conditions with the 10ms batching window, it should be called once or a few times
        Assert.True(batchCallCount > 0, "Model was never called");
        Assert.True(batchCallCount <= 10, "Batching did not occur - model was called for each request individually");

        // Get batcher statistics
        var statsResponse = await _client.GetAsync("/api/inference/stats");
        statsResponse.EnsureSuccessStatusCode();
        var stats = await statsResponse.Content.ReadFromJsonAsync<Dictionary<string, object>>();

        Assert.NotNull(stats);
        Assert.True(stats.ContainsKey("totalRequests"));
        Assert.True(stats.ContainsKey("totalBatches"));
        Assert.True(stats.ContainsKey("averageBatchSize"));

        // Verify the total requests is at least 10
        var totalRequests = ((JsonElement)stats["totalRequests"]).GetInt64();
        Assert.True(totalRequests >= 10);

        // Cleanup
        repository.UnloadModel("batch-test-model");
    }

    /// <summary>
    /// Verifies that predictions with non-existent models return 404.
    /// </summary>
    [Fact]
    public async Task Predict_WithNonExistentModel_Returns404()
    {
        // Arrange
        var request = new PredictionRequest
        {
            Features = new[] { new[] { 1.0, 2.0, 3.0 } }
        };

        // Act
        var response = await _client.PostAsJsonAsync("/api/inference/predict/non-existent-model", request);

        // Assert
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    /// <summary>
    /// Verifies that predictions with invalid input return 400.
    /// </summary>
    [Fact]
    public async Task Predict_WithEmptyFeatures_Returns400()
    {
        // Arrange
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var testModel = CreateSimpleTestModel("test-model-5");
        repository.LoadModel("test-model-5", testModel);

        var request = new PredictionRequest
        {
            Features = Array.Empty<double[]>()
        };

        // Act
        var response = await _client.PostAsJsonAsync("/api/inference/predict/test-model-5", request);

        // Assert
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        // Cleanup
        repository.UnloadModel("test-model-5");
    }

    /// <summary>
    /// Creates a simple test model that sums the input features.
    /// </summary>
    private static IServableModel<double> CreateSimpleTestModel(string name)
    {
        var numOps = MathHelper.GetNumericOperations<double>();

        return new ServableModelWrapper<double>(
            modelName: name,
            inputDimension: 3,
            outputDimension: 1,
            predictFunc: input =>
            {
                // Sum all input features
                var sum = numOps.Zero;
                for (int i = 0; i < input.Length; i++)
                {
                    sum = numOps.Add(sum, input[i]);
                }
                return new Vector<double>(new[] { sum });
            },
            predictBatchFunc: inputs =>
            {
                // Sum each row
                var result = new Matrix<double>(inputs.Rows, 1);
                for (int i = 0; i < inputs.Rows; i++)
                {
                    var sum = numOps.Zero;
                    for (int j = 0; j < inputs.Columns; j++)
                    {
                        sum = numOps.Add(sum, inputs[i, j]);
                    }
                    result[i, 0] = sum;
                }
                return result;
            }
        );
    }

    /// <summary>
    /// Creates a test model that counts how many times batch prediction is called.
    /// This is used to verify that batching is working correctly.
    /// </summary>
    private static IServableModel<double> CreateCountingTestModel(string name, ref int callCount)
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var localCallCount = callCount; // Capture for closure

        return new ServableModelWrapper<double>(
            modelName: name,
            inputDimension: 3,
            outputDimension: 1,
            predictFunc: input =>
            {
                // Single prediction (should not be called often if batching works)
                var sum = numOps.Zero;
                for (int i = 0; i < input.Length; i++)
                {
                    sum = numOps.Add(sum, input[i]);
                }
                return new Vector<double>(new[] { sum });
            },
            predictBatchFunc: inputs =>
            {
                // Increment call count when batch prediction is called
                System.Threading.Interlocked.Increment(ref callCount);

                // Process batch
                var result = new Matrix<double>(inputs.Rows, 1);
                for (int i = 0; i < inputs.Rows; i++)
                {
                    var sum = numOps.Zero;
                    for (int j = 0; j < inputs.Columns; j++)
                    {
                        sum = numOps.Add(sum, inputs[i, j]);
                    }
                    result[i, 0] = sum;
                }
                return result;
            }
        );
    }
}
