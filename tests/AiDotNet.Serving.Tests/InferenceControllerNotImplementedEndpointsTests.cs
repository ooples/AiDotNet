using System.Net;
using System.Reflection;
using System.Text;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Serialization;
using Xunit;

namespace AiDotNet.Serving.Tests;

[Collection("ServingIntegrationTests")]
public class InferenceControllerNotImplementedEndpointsTests : IClassFixture<WebApplicationFactory<Program>>, IAsyncLifetime
{
    private static readonly JsonSerializerSettings JsonSettings = new()
    {
        Converters = { new StringEnumConverter(new CamelCaseNamingStrategy(), allowIntegerValues: false) }
    };

    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;
    private readonly List<string> _loadedModels = new();

    public InferenceControllerNotImplementedEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory;
        _client = _factory.CreateClient();
    }

    private static async Task<T?> ReadAsJsonAsync<T>(HttpContent content)
    {
        var stringContent = await content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<T>(stringContent, JsonSettings);
    }

    private async Task<HttpResponseMessage> PostAsJsonAsync<T>(string requestUri, T value)
    {
        var content = new StringContent(JsonConvert.SerializeObject(value, JsonSettings), Encoding.UTF8, "application/json");
        return await _client.PostAsync(requestUri, content);
    }

    public Task InitializeAsync()
    {
        CleanupLoadedModels();
        return Task.CompletedTask;
    }

    public Task DisposeAsync()
    {
        CleanupLoadedModels();
        return Task.CompletedTask;
    }

    [Fact]
    public async Task GenerateWithSpeculativeDecoding_WhenRequestInvalid_Returns400()
    {
        var response = await PostAsJsonAsync(
            "/api/inference/generate/any-model",
            new SpeculativeDecodingRequest { InputTokens = Array.Empty<int>() });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var payload = await ReadAsJsonAsync<SpeculativeDecodingResponse>(response.Content);
        Assert.NotNull(payload);
        Assert.Equal("InputTokens array is required and cannot be empty", payload!.Error);
    }

    [Fact]
    public async Task GenerateWithSpeculativeDecoding_WhenModelMissing_Returns404()
    {
        var response = await PostAsJsonAsync(
            "/api/inference/generate/missing-model",
            new SpeculativeDecodingRequest { InputTokens = new[] { 1 }, MaxNewTokens = 1, Temperature = 1.0, NumDraftTokens = 1 });

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);

        var payload = await ReadAsJsonAsync<SpeculativeDecodingResponse>(response.Content);
        Assert.NotNull(payload);
        Assert.Contains("missing-model", payload!.Error ?? string.Empty, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GenerateWithSpeculativeDecoding_WhenModelExists_Returns501()
    {
        LoadSimpleModel("spec-decoding-model", enableBatching: true);

        var response = await PostAsJsonAsync(
            "/api/inference/generate/spec-decoding-model",
            new SpeculativeDecodingRequest { InputTokens = new[] { 1 }, MaxNewTokens = 1, Temperature = 1.0, NumDraftTokens = 1 });

        Assert.Equal(HttpStatusCode.NotImplemented, response.StatusCode);

        var payload = await ReadAsJsonAsync<SpeculativeDecodingResponse>(response.Content);
        Assert.NotNull(payload);
        Assert.Equal("Speculative decoding is not available via the REST API in the current version.", payload!.Error);
    }

    [Fact]
    public async Task FineTuneWithLoRA_WhenRequestInvalid_Returns400()
    {
        var response = await PostAsJsonAsync(
            "/api/inference/finetune/lora",
            new LoRAFineTuneRequest { ModelName = " " });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var payload = await ReadAsJsonAsync<LoRAFineTuneResponse>(response.Content);
        Assert.NotNull(payload);
        Assert.False(payload!.Success);
        Assert.Equal("ModelName is required", payload.Error);
    }

    [Fact]
    public async Task FineTuneWithLoRA_WhenModelMissing_Returns404()
    {
        var response = await PostAsJsonAsync(
            "/api/inference/finetune/lora",
            new LoRAFineTuneRequest
            {
                ModelName = "missing-model",
                TrainingFeatures = new[] { new[] { 1.0, 2.0 } },
                TrainingLabels = new[] { new[] { 3.0 } }
            });

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);

        var payload = await ReadAsJsonAsync<LoRAFineTuneResponse>(response.Content);
        Assert.NotNull(payload);
        Assert.False(payload!.Success);
        Assert.Contains("missing-model", payload.Error ?? string.Empty, StringComparison.Ordinal);
    }

    [Fact]
    public async Task FineTuneWithLoRA_WhenModelExists_Returns501()
    {
        LoadSimpleModel("lora-model", enableBatching: true);

        var response = await PostAsJsonAsync(
            "/api/inference/finetune/lora",
            new LoRAFineTuneRequest
            {
                ModelName = "lora-model",
                TrainingFeatures = new[] { new[] { 1.0, 2.0 } },
                TrainingLabels = new[] { new[] { 3.0 } }
            });

        Assert.Equal(HttpStatusCode.NotImplemented, response.StatusCode);

        var payload = await ReadAsJsonAsync<LoRAFineTuneResponse>(response.Content);
        Assert.NotNull(payload);
        Assert.False(payload!.Success);
        Assert.Equal("LoRA fine-tuning is not available via the REST API in the current version.", payload.Error);
    }

    [Fact]
    public async Task Predict_WhenBatchingDisabledAndTooManyItems_Returns413()
    {
        LoadSimpleModel("unbatched-model", enableBatching: false);

        var features = new double[1001][];
        for (int i = 0; i < features.Length; i++)
        {
            features[i] = new[] { 1.0 };
        }

        var response = await PostAsJsonAsync(
            "/api/inference/predict/unbatched-model",
            new PredictionRequest { Features = features });

        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);

        var payload = await ReadAsJsonAsync<Dictionary<string, string>>(response.Content);
        Assert.NotNull(payload);
        Assert.True(payload!.TryGetValue("error", out var error));
        Assert.Contains("batching is disabled", error ?? string.Empty, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void LoRAFineTuneRequest_Validate_CoversAllErrorBranches()
    {
        Assert.Equal("ModelName is required", Validate(new LoRAFineTuneRequest { ModelName = " " }));

        Assert.Equal(
            "TrainingFeatures array is required and cannot be empty",
            Validate(new LoRAFineTuneRequest { ModelName = "m", TrainingFeatures = Array.Empty<double[]>(), TrainingLabels = new[] { new[] { 1.0 } } }));

        Assert.Equal(
            "TrainingLabels array is required and cannot be empty",
            Validate(new LoRAFineTuneRequest { ModelName = "m", TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = Array.Empty<double[]>() }));

        Assert.Equal(
            "TrainingFeatures and TrainingLabels must have the same length",
            Validate(new LoRAFineTuneRequest { ModelName = "m", TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = new[] { new[] { 1.0 }, new[] { 2.0 } } }));

        Assert.Equal(
            "TrainingFeatures rows must be non-empty",
            Validate(new LoRAFineTuneRequest { ModelName = "m", TrainingFeatures = new[] { Array.Empty<double>() }, TrainingLabels = new[] { new[] { 1.0 } } }));

        Assert.Equal(
            "All TrainingFeatures rows must have the same length",
            Validate(new LoRAFineTuneRequest { ModelName = "m", TrainingFeatures = new[] { new[] { 1.0 }, new[] { 1.0, 2.0 } }, TrainingLabels = new[] { new[] { 1.0 }, new[] { 2.0 } } }));

        Assert.Equal(
            "TrainingLabels rows must be non-empty",
            Validate(new LoRAFineTuneRequest { ModelName = "m", TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = new[] { Array.Empty<double>() } }));

        Assert.Equal(
            "All TrainingLabels rows must have the same length",
            Validate(new LoRAFineTuneRequest { ModelName = "m", TrainingFeatures = new[] { new[] { 1.0 }, new[] { 2.0 } }, TrainingLabels = new[] { new[] { 1.0 }, new[] { 2.0, 3.0 } } }));

        Assert.Equal(
            "Rank must be greater than 0",
            Validate(new LoRAFineTuneRequest { ModelName = "m", Rank = 0, TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = new[] { new[] { 1.0 } } }));

        Assert.Equal(
            "Alpha must be greater than 0",
            Validate(new LoRAFineTuneRequest { ModelName = "m", Alpha = 0.0, TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = new[] { new[] { 1.0 } } }));

        Assert.Equal(
            "LearningRate must be greater than 0",
            Validate(new LoRAFineTuneRequest { ModelName = "m", LearningRate = 0.0, TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = new[] { new[] { 1.0 } } }));

        Assert.Equal(
            "Epochs must be greater than 0",
            Validate(new LoRAFineTuneRequest { ModelName = "m", Epochs = 0, TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = new[] { new[] { 1.0 } } }));

        Assert.Equal(
            "BatchSize must be greater than 0",
            Validate(new LoRAFineTuneRequest { ModelName = "m", BatchSize = 0, TrainingFeatures = new[] { new[] { 1.0 } }, TrainingLabels = new[] { new[] { 1.0 } } }));

        Assert.Equal(
            "SavePath is required when SaveModel is true",
            Validate(new LoRAFineTuneRequest
            {
                ModelName = "m",
                TrainingFeatures = new[] { new[] { 1.0 } },
                TrainingLabels = new[] { new[] { 1.0 } },
                SaveModel = true
            }));

        Assert.Null(Validate(new LoRAFineTuneRequest
        {
            ModelName = "m",
            TrainingFeatures = new[] { new[] { 1.0, 2.0 } },
            TrainingLabels = new[] { new[] { 3.0 } },
            SaveModel = true,
            SavePath = "out.model"
        }));
    }

    [Fact]
    public void SpeculativeDecodingRequest_Validate_CoversAllErrorBranches()
    {
        Assert.Equal(
            "InputTokens array is required and cannot be empty",
            Validate(new SpeculativeDecodingRequest { InputTokens = Array.Empty<int>() }));

        Assert.Equal(
            "MaxNewTokens must be greater than 0",
            Validate(new SpeculativeDecodingRequest { InputTokens = new[] { 1 }, MaxNewTokens = 0 }));

        Assert.Equal(
            "Temperature must be greater than 0",
            Validate(new SpeculativeDecodingRequest { InputTokens = new[] { 1 }, Temperature = 0.0 }));

        Assert.Equal(
            "NumDraftTokens must be greater than 0",
            Validate(new SpeculativeDecodingRequest { InputTokens = new[] { 1 }, NumDraftTokens = 0 }));

        Assert.Equal(
            "TreeBranchFactor must be greater than 0 when UseTreeSpeculation is true",
            Validate(new SpeculativeDecodingRequest { InputTokens = new[] { 1 }, UseTreeSpeculation = true, TreeBranchFactor = 0 }));

        Assert.Equal(
            "MaxTreeDepth must be greater than 0 when UseTreeSpeculation is true",
            Validate(new SpeculativeDecodingRequest { InputTokens = new[] { 1 }, UseTreeSpeculation = true, TreeBranchFactor = 2, MaxTreeDepth = 0 }));

        Assert.Null(Validate(new SpeculativeDecodingRequest
        {
            InputTokens = new[] { 1 },
            MaxNewTokens = 1,
            Temperature = 1.0,
            NumDraftTokens = 1,
            UseTreeSpeculation = true,
            TreeBranchFactor = 2,
            MaxTreeDepth = 2
        }));
    }

    private static string? Validate(object request)
    {
        var method = request.GetType().GetMethod("Validate", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return (string?)method!.Invoke(request, null);
    }

    private void LoadSimpleModel(string name, bool enableBatching)
    {
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        var model = new ServableModelWrapper<double>(
            name,
            inputDimension: 1,
            outputDimension: 1,
            predictFunc: input => new Vector<double>(new[] { input[0] }),
            predictBatchFunc: inputs =>
            {
                var result = new Matrix<double>(inputs.Rows, 1);
                for (int row = 0; row < inputs.Rows; row++)
                {
                    result[row, 0] = inputs[row, 0];
                }
                return result;
            },
            enableBatching: enableBatching);

        var loaded = repository.LoadModel(name, model);
        Assert.True(loaded, $"Failed to load model '{name}' for test setup");
        _loadedModels.Add(name);
    }

    private void CleanupLoadedModels()
    {
        using var scope = _factory.Services.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<IModelRepository>();

        foreach (var modelName in _loadedModels)
        {
            repository.UnloadModel(modelName);
        }

        _loadedModels.Clear();
    }
}
