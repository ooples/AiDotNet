using System.Net;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Integration tests for the AiDotNet Serving API.
/// These tests verify the end-to-end functionality of the model serving framework,
/// including model management, request batching, and inference endpoints.
/// </summary>
[Collection("ServingIntegrationTests")]
public class ServingIntegrationTests : IClassFixture<WebApplicationFactory<Program>>, IAsyncLifetime
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase, allowIntegerValues: false) }
    };

    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public ServingIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory;
        _client = _factory.CreateClient();
    }

    /// <summary>
    /// Initializes the test (called before each test method).
    /// Cleans up any models left over from previous tests to ensure proper test isolation.
    /// </summary>
    public Task InitializeAsync()
    {
        // Clean up any models left over from previous tests (ensures isolation even if DisposeAsync failed)
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var models = repository.GetAllModelInfo();
        foreach (var model in models)
        {
            repository.UnloadModel(model.Name);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Cleans up after each test by unloading all models from the singleton repository.
    /// This ensures test isolation even though IModelRepository is a singleton.
    /// </summary>
    public Task DisposeAsync()
    {
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        // Unload all models to ensure test isolation
        var models = repository.GetAllModelInfo();
        foreach (var model in models)
        {
            repository.UnloadModel(model.Name);
        }

        return Task.CompletedTask;
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
        var models = await response.Content.ReadFromJsonAsync<List<ModelInfo>>(JsonOptions);
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
        var models = await response.Content.ReadFromJsonAsync<List<ModelInfo>>(JsonOptions);
        Assert.NotNull(models);
        Assert.Single(models);
        Assert.Equal("test-model-1", models[0].Name);
        Assert.Equal(NumericType.Double, models[0].NumericType);
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
        var modelInfo = await response.Content.ReadFromJsonAsync<ModelInfo>(JsonOptions);
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
        var response = await _client.PostAsJsonAsync("/api/inference/predict/test-model-4", request, JsonOptions);

        // Assert
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<PredictionResponse>(JsonOptions);
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
    /// Verifies that serving can route to a pre-loaded model variant via an adapter header (Multi-LoRA MVP).
    /// </summary>
    [Fact]
    public async Task Predict_WithAdapterHeader_RoutesToModelVariant()
    {
        // Arrange
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var baseName = "test-model-variant";
        var adapterId = "adapterA";
        var variantName = $"{baseName}__{adapterId}";

        repository.LoadModel(baseName, CreateSimpleTestModel(baseName));

        // Variant model returns (sum + 100) so we can detect routing.
        var numOps = MathHelper.GetNumericOperations<double>();
        var variant = new ServableModelWrapper<double>(
            modelName: variantName,
            inputDimension: 3,
            outputDimension: 1,
            predictFunc: input =>
            {
                var sum = numOps.Zero;
                for (int i = 0; i < input.Length; i++)
                {
                    sum = numOps.Add(sum, input[i]);
                }
                return new Vector<double>(new[] { sum + 100.0 });
            });
        repository.LoadModel(variantName, variant);

        var request = new PredictionRequest
        {
            Features = new[] { new[] { 1.0, 2.0, 3.0 } },
            RequestId = "test-request-variant"
        };

        // Act
        var message = new HttpRequestMessage(HttpMethod.Post, $"/api/inference/predict/{baseName}")
        {
            Content = JsonContent.Create(request)
        };
        message.Headers.Add("X-AiDotNet-Lora", adapterId);

        var response = await _client.SendAsync(message);

        // Assert
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<PredictionResponse>(JsonOptions);
        Assert.NotNull(result);
        Assert.Equal("test-request-variant", result.RequestId);
        Assert.NotNull(result.Predictions);
        Assert.Single(result.Predictions);
        Assert.Single(result.Predictions[0]);
        Assert.Equal(106.0, result.Predictions[0][0], 5);

        // Cleanup
        repository.UnloadModel(baseName);
        repository.UnloadModel(variantName);
    }

    /// <summary>
    /// Critical test: Verifies that batch processing works correctly.
    /// This test ensures that multiple concurrent requests are batched together
    /// and the model is called once with the full batch.
    /// Note: This test now uses Channel-based batching which is reliable in CI environments.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Predict_WithConcurrentRequests_ProcessesAsBatch()
    {
        // Arrange
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var batchCallCount = new StrongBox<int>(0);
        var testModel = CreateCountingTestModel("batch-test-model", batchCallCount);
        repository.LoadModel("batch-test-model", testModel);

        // Create 10 prediction requests
        var requests = Enumerable.Range(0, 10).Select(i => new PredictionRequest
        {
            Features = new[] { new[] { (double)i, (double)(i + 1), (double)(i + 2) } },
            RequestId = $"batch-request-{i}"
        }).ToArray();

        // Act: Send all requests concurrently with a timeout to prevent hanging
        // Using 90 seconds to allow for slow CI environments
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(90));
        var tasks = requests.Select(req =>
            _client.PostAsJsonAsync("/api/inference/predict/batch-test-model", req, JsonOptions, cts.Token)
        ).ToArray();

        var responses = await Task.WhenAll(tasks);

        // Assert: All requests should succeed
        foreach (var response in responses)
        {
            response.EnsureSuccessStatusCode();
        }

        // Wait for batch processing to complete with polling instead of fixed delay
        // This prevents flakiness from race conditions
        // Increased timeout for CI environments which can be slower
        var maxWaitMs = 5000;
        var pollIntervalMs = 10;
        var waited = 0;
        while (batchCallCount.Value == 0 && waited < maxWaitMs)
        {
            await Task.Delay(pollIntervalMs);
            waited += pollIntervalMs;
        }

        // Verify that batch processing occurred
        // The model should have been called fewer times than the number of requests
        // In ideal conditions with the 10ms batching window, it should be called once or a few times
        Assert.True(batchCallCount.Value > 0, "Model was never called");
        // Ensure batching actually reduced the number of calls - must be less than total requests
        Assert.True(batchCallCount.Value < requests.Length, "Batching did not occur - model was called for each request individually");

        // Get batcher statistics
        // Note: Using test-level timeout instead of HTTP-level timeout here
        // since the batch requests may have consumed most of the CTS timeout
        var statsResponse = await _client.GetAsync("/api/inference/stats");
        statsResponse.EnsureSuccessStatusCode();
        var stats = await statsResponse.Content.ReadFromJsonAsync<Dictionary<string, object>>(JsonOptions);

        Assert.NotNull(stats);
        Assert.True(stats.TryGetValue("totalRequests", out var totalRequestsObj));
        Assert.True(stats.TryGetValue("totalBatches", out _));
        Assert.True(stats.TryGetValue("averageBatchSize", out _));

        // Verify the total requests is at least 10
        var totalRequests = ((JsonElement)totalRequestsObj).GetInt64();
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
        var response = await _client.PostAsJsonAsync("/api/inference/predict/non-existent-model", request, JsonOptions);

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
        var response = await _client.PostAsJsonAsync("/api/inference/predict/test-model-5", request, JsonOptions);

        // Assert
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        // Cleanup
        repository.UnloadModel("test-model-5");
    }

    [Fact]
    public async Task DownloadModelArtifact_FreeTier_ReturnsForbidden()
    {
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();
        var servingOptions = scope.ServiceProvider.GetRequiredService<IOptions<ServingOptions>>().Value;

        var modelsRoot = Path.GetFullPath(servingOptions.ModelDirectory);
        Directory.CreateDirectory(modelsRoot);

        var artifactPath = Path.Combine(modelsRoot, "artifact-free.bin");
        File.WriteAllBytes(artifactPath, new byte[] { 1, 2, 3, 4 });

        var modelName = "artifact-model-free";
        repository.LoadModel(modelName, CreateSimpleTestModel(modelName), sourcePath: artifactPath);

        var request = new HttpRequestMessage(HttpMethod.Get, $"/api/models/{modelName}/artifact");

        var response = await _client.SendAsync(request);
        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);

        repository.UnloadModel(modelName);
        File.Delete(artifactPath);
    }

    [Fact]
    public async Task DownloadModelArtifact_ProTier_ReturnsEncryptedArtifact_AndKeyReleaseSucceeds()
    {
        var proApiKey = await CreateApiKeyAsync(SubscriptionTier.Pro);

        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();
        var servingOptions = scope.ServiceProvider.GetRequiredService<IOptions<ServingOptions>>().Value;

        var modelsRoot = Path.GetFullPath(servingOptions.ModelDirectory);
        Directory.CreateDirectory(modelsRoot);

        var artifactPath = Path.Combine(modelsRoot, "artifact-pro.bin");
        var expected = new byte[] { 10, 20, 30, 40, 50 };
        File.WriteAllBytes(artifactPath, expected);

        var modelName = "artifact-model-pro";
        repository.LoadModel(modelName, CreateSimpleTestModel(modelName), sourcePath: artifactPath);

        var request = new HttpRequestMessage(HttpMethod.Get, $"/api/models/{modelName}/artifact");
        request.Headers.Add("X-AiDotNet-ApiKey", proApiKey);

        var response = await _client.SendAsync(request);
        response.EnsureSuccessStatusCode();

        Assert.True(response.Headers.TryGetValues("X-AiDotNet-Artifact-Encrypted", out var encryptedHeader));
        Assert.Equal("true", encryptedHeader!.First());

        var encryptedBytes = await response.Content.ReadAsByteArrayAsync();
        Assert.NotEmpty(encryptedBytes);

        var keyRequest = new HttpRequestMessage(HttpMethod.Post, $"/api/models/{modelName}/artifact/key");
        keyRequest.Headers.Add("X-AiDotNet-ApiKey", proApiKey);
        keyRequest.Content = new StringContent("null", Encoding.UTF8, "application/json");

        var keyResponse = await _client.SendAsync(keyRequest);
        keyResponse.EnsureSuccessStatusCode();

        var keyPayload = await keyResponse.Content.ReadFromJsonAsync<ModelArtifactKeyResponse>(JsonOptions);
        Assert.NotNull(keyPayload);
        Assert.False(string.IsNullOrWhiteSpace(keyPayload!.KeyId));
        Assert.False(string.IsNullOrWhiteSpace(keyPayload.KeyBase64));
        Assert.False(string.IsNullOrWhiteSpace(keyPayload.NonceBase64));

        var decryptedBytes = DecryptAesGcmArtifact(modelName, encryptedBytes, keyPayload);
        Assert.Equal(expected, decryptedBytes);

        repository.UnloadModel(modelName);
        File.Delete(artifactPath);
    }

    [Fact]
    public async Task DownloadModelArtifact_EnterpriseTier_ReturnsEncryptedArtifact_AndKeyReleaseSucceeds()
    {
        var enterpriseApiKey = await CreateApiKeyAsync(SubscriptionTier.Enterprise);

        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();
        var servingOptions = scope.ServiceProvider.GetRequiredService<IOptions<ServingOptions>>().Value;

        var modelsRoot = Path.GetFullPath(servingOptions.ModelDirectory);
        Directory.CreateDirectory(modelsRoot);

        var artifactPath = Path.Combine(modelsRoot, "artifact-ent.bin");
        var expected = Enumerable.Range(0, 128).Select(i => (byte)i).ToArray();
        File.WriteAllBytes(artifactPath, expected);

        var modelName = "artifact-model-ent";
        repository.LoadModel(modelName, CreateSimpleTestModel(modelName), sourcePath: artifactPath);

        var artifactRequest = new HttpRequestMessage(HttpMethod.Get, $"/api/models/{modelName}/artifact");
        artifactRequest.Headers.Add("X-AiDotNet-ApiKey", enterpriseApiKey);
        var artifactResponse = await _client.SendAsync(artifactRequest);

        artifactResponse.EnsureSuccessStatusCode();
        Assert.True(artifactResponse.Headers.TryGetValues("X-AiDotNet-Artifact-Encrypted", out var encryptedHeader));
        Assert.Equal("true", encryptedHeader!.First());

        var encryptedBytes = await artifactResponse.Content.ReadAsByteArrayAsync();
        Assert.NotEmpty(encryptedBytes);

        var keyRequest = new HttpRequestMessage(HttpMethod.Post, $"/api/models/{modelName}/artifact/key");
        keyRequest.Headers.Add("X-AiDotNet-ApiKey", enterpriseApiKey);
        keyRequest.Content = JsonContent.Create(new AttestationEvidence
        {
            Platform = "Windows",
            TeeType = "Test",
            Nonce = "nonce",
            AttestationToken = "token"
        });

        var keyResponse = await _client.SendAsync(keyRequest);
        keyResponse.EnsureSuccessStatusCode();

        var keyPayload = await keyResponse.Content.ReadFromJsonAsync<ModelArtifactKeyResponse>(JsonOptions);
        Assert.NotNull(keyPayload);
        Assert.False(string.IsNullOrWhiteSpace(keyPayload!.KeyId));
        Assert.False(string.IsNullOrWhiteSpace(keyPayload.KeyBase64));
        Assert.False(string.IsNullOrWhiteSpace(keyPayload.NonceBase64));

        var decryptedBytes = DecryptAesGcmArtifact(modelName, encryptedBytes, keyPayload);
        Assert.Equal(expected, decryptedBytes);

        repository.UnloadModel(modelName);
        File.Delete(artifactPath);

        var protectedDir = Path.Combine(modelsRoot, ".protected");
        var protectedPath = Path.Combine(protectedDir, $"{modelName}.aidn.enc");
        if (File.Exists(protectedPath))
        {
            File.Delete(protectedPath);
        }
    }

    private async Task<string> CreateApiKeyAsync(SubscriptionTier tier)
    {
        using var scope = _factory.Services.CreateScope();
        var apiKeys = scope.ServiceProvider.GetRequiredService<IApiKeyService>();

        var created = await apiKeys.CreateAsync(new ApiKeyCreateRequest
        {
            Name = $"serving-tests-{tier}-{Guid.NewGuid():N}",
            Tier = tier
        });

        return created.ApiKey;
    }

    private static byte[] DecryptAesGcmArtifact(string aadText, byte[] protectedFileBytes, ModelArtifactKeyResponse keyResponse)
    {
        if (string.IsNullOrWhiteSpace(aadText))
        {
            throw new ArgumentException("AAD text is required.", nameof(aadText));
        }

        if (protectedFileBytes == null)
        {
            throw new ArgumentNullException(nameof(protectedFileBytes));
        }

        if (keyResponse == null)
        {
            throw new ArgumentNullException(nameof(keyResponse));
        }

        var key = Convert.FromBase64String(keyResponse.KeyBase64);
        var expectedNonce = Convert.FromBase64String(keyResponse.NonceBase64);
        var aad = Encoding.UTF8.GetBytes(aadText);

        const int magicLen = 4;
        const int versionLen = 1;
        const int nonceLen = 12;
        const int tagLen = 16;
        var minLen = magicLen + versionLen + nonceLen + tagLen;

        if (protectedFileBytes.Length < minLen)
        {
            throw new InvalidOperationException("Protected artifact is too small.");
        }

        var magic = protectedFileBytes.AsSpan(0, magicLen);
        if (!magic.SequenceEqual("AIDN"u8))
        {
            throw new InvalidOperationException("Protected artifact has an invalid magic header.");
        }

        var version = protectedFileBytes[magicLen];
        if (version != 1)
        {
            throw new InvalidOperationException($"Unsupported protected artifact version: {version}.");
        }

        var offset = magicLen + versionLen;
        var nonce = protectedFileBytes.AsSpan(offset, nonceLen).ToArray();
        offset += nonceLen;

        if (expectedNonce.Length == nonceLen)
        {
            Assert.Equal(expectedNonce, nonce);
        }

        var tag = protectedFileBytes.AsSpan(offset, tagLen).ToArray();
        offset += tagLen;

        var ciphertext = protectedFileBytes.AsSpan(offset).ToArray();
        var plaintext = new byte[ciphertext.Length];

        try
        {
            using var aes = new AesGcm(key, tagLen);
            aes.Decrypt(nonce, ciphertext, tag, plaintext, aad);
            return plaintext;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(key);
            CryptographicOperations.ZeroMemory(expectedNonce);
            CryptographicOperations.ZeroMemory(aad);
            CryptographicOperations.ZeroMemory(nonce);
            CryptographicOperations.ZeroMemory(tag);
            CryptographicOperations.ZeroMemory(ciphertext);
        }
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
    private static IServableModel<double> CreateCountingTestModel(string name, StrongBox<int> callCount)
    {
        var numOps = MathHelper.GetNumericOperations<double>();

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
                System.Threading.Interlocked.Increment(ref callCount.Value);

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
