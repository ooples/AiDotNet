using System.Net;
using System.Security.Cryptography;
using System.Text;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Models.Federated;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Serialization;
using Xunit;

namespace AiDotNet.Serving.Tests;

[Collection("ServingIntegrationTests")]
public class FederatedCoordinatorIntegrationTests : IClassFixture<WebApplicationFactory<Program>>, IAsyncLifetime
{
    private const string ApiKeyHeaderName = "X-AiDotNet-ApiKey";

    private static readonly JsonSerializerSettings JsonSettings = new()
    {
        Converters = { new StringEnumConverter(new CamelCaseNamingStrategy(), allowIntegerValues: false) }
    };

    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;
    private readonly List<string> _createdModelFiles = new();
    private readonly List<string> _loadedModels = new();

    public FederatedCoordinatorIntegrationTests(WebApplicationFactory<Program> factory)
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

    private async Task<T?> GetAsJsonAsync<T>(string requestUri)
    {
        var response = await _client.GetAsync(requestUri);
        response.EnsureSuccessStatusCode();
        var content = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<T>(content, JsonSettings);
    }

    public Task InitializeAsync()
    {
        CleanupLoadedModels();
        CleanupCreatedFiles();
        return Task.CompletedTask;
    }

    public Task DisposeAsync()
    {
        CleanupLoadedModels();
        CleanupCreatedFiles();
        return Task.CompletedTask;
    }

    [Fact]
    public async Task CreateRun_WhenModelDoesNotExist_Returns404()
    {
        var response = await PostAsJsonAsync("/api/federated/runs", new CreateFederatedRunRequest
        {
            ModelName = "non-existent-model",
            Options = new FederatedLearningOptions(),
            MinClientUpdatesPerRound = 1
        });

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task JoinRun_EnterpriseTier_WithNullAttestation_Returns403()
    {
        var modelName = "fed-join-enterprise";
        await LoadTrainedVectorModelAsync(modelName);

        var create = await PostAsJsonAsync("/api/federated/runs", new CreateFederatedRunRequest
        {
            ModelName = modelName,
            Options = new FederatedLearningOptions { AggregationStrategy = FederatedAggregationStrategy.FedAvg },
            MinClientUpdatesPerRound = 1
        });
        create.EnsureSuccessStatusCode();
        var createResponse = await ReadAsJsonAsync<CreateFederatedRunResponse>(create.Content);
        Assert.NotNull(createResponse);

        var enterpriseApiKey = await CreateApiKeyAsync(SubscriptionTier.Enterprise);
        SetApiKey(enterpriseApiKey);
        var join = await PostAsJsonAsync($"/api/federated/runs/{createResponse.RunId}/join", new JoinFederatedRunRequest
        {
            ClientId = null,
            Attestation = null!
        });

        if (join.StatusCode != HttpStatusCode.Forbidden)
        {
            var body = await join.Content.ReadAsStringAsync();
            throw new InvalidOperationException($"Expected 403 Forbidden, got {(int)join.StatusCode}. Body: {body}");
        }
    }

    [Fact]
    public async Task FederatedRun_Lifecycle_FedAvg_AggregatesAndAdvancesRound()
    {
        var modelName = "fed-lifecycle";
        await LoadTrainedVectorModelAsync(modelName);

        var proApiKey = await CreateApiKeyAsync(SubscriptionTier.Pro);
        var enterpriseApiKey = await CreateApiKeyAsync(SubscriptionTier.Enterprise);
        SetApiKey(null);
        var create = await PostAsJsonAsync("/api/federated/runs", new CreateFederatedRunRequest
        {
            ModelName = modelName,
            Options = new FederatedLearningOptions { AggregationStrategy = FederatedAggregationStrategy.FedAvg },
            MinClientUpdatesPerRound = 2
        });
        if (!create.IsSuccessStatusCode)
        {
            var body = await create.Content.ReadAsStringAsync();
            throw new InvalidOperationException($"CreateRun failed (HTTP {(int)create.StatusCode}): {body}");
        }
        var createResponse = await ReadAsJsonAsync<CreateFederatedRunResponse>(create.Content);
        Assert.NotNull(createResponse);
        Assert.True(createResponse.ParameterCount > 0);

        var join0 = await PostAsJsonAsync($"/api/federated/runs/{createResponse.RunId}/join", new JoinFederatedRunRequest
        {
            Attestation = CreateAttestationEvidence()
        });
        join0.EnsureSuccessStatusCode();
        var join0Response = await ReadAsJsonAsync<JoinFederatedRunResponse>(join0.Content);
        Assert.NotNull(join0Response);

        var join1 = await PostAsJsonAsync($"/api/federated/runs/{createResponse.RunId}/join", new JoinFederatedRunRequest
        {
            Attestation = CreateAttestationEvidence()
        });
        join1.EnsureSuccessStatusCode();
        var join1Response = await ReadAsJsonAsync<JoinFederatedRunResponse>(join1.Content);
        Assert.NotNull(join1Response);

        var initial = await GetAsJsonAsync<FederatedRunParametersResponse>(
            $"/api/federated/runs/{createResponse.RunId}/clients/{join0Response.ClientId}/parameters");
        Assert.NotNull(initial);
        Assert.Equal(0, initial.RoundNumber);
        Assert.Equal(createResponse.ParameterCount, initial.ParameterCount);
        Assert.Equal(createResponse.ParameterCount, initial.Parameters.Length);

        var ones = Enumerable.Repeat(1.0, createResponse.ParameterCount).ToArray();
        var threes = Enumerable.Repeat(3.0, createResponse.ParameterCount).ToArray();

        var update0 = await PostAsJsonAsync($"/api/federated/runs/{createResponse.RunId}/updates", new SubmitFederatedUpdateRequest
        {
            ClientId = join0Response.ClientId,
            RoundNumber = 0,
            ClientWeight = 1.0,
            Parameters = ones
        });
        update0.EnsureSuccessStatusCode();

        var update1 = await PostAsJsonAsync($"/api/federated/runs/{createResponse.RunId}/updates", new SubmitFederatedUpdateRequest
        {
            ClientId = join1Response.ClientId,
            RoundNumber = 0,
            ClientWeight = 1.0,
            Parameters = threes
        });
        update1.EnsureSuccessStatusCode();

        var aggregate = await PostAsJsonAsync($"/api/federated/runs/{createResponse.RunId}/aggregate", new { });
        aggregate.EnsureSuccessStatusCode();
        var aggregateResponse = await ReadAsJsonAsync<AggregateFederatedRoundResponse>(aggregate.Content);
        Assert.NotNull(aggregateResponse);
        Assert.Equal(1, aggregateResponse.NewCurrentRound);
        Assert.Equal(2, aggregateResponse.AggregatedClientCount);

        var after = await GetAsJsonAsync<FederatedRunParametersResponse>(
            $"/api/federated/runs/{createResponse.RunId}/clients/{join0Response.ClientId}/parameters");
        Assert.NotNull(after);
        Assert.Equal(1, after.RoundNumber);
        Assert.Equal(createResponse.ParameterCount, after.Parameters.Length);
        Assert.Equal(2.0, after.Parameters[0], precision: 6);
        Assert.Equal(2.0, after.Parameters[after.Parameters.Length - 1], precision: 6);

        // Option A (Free) forbids artifact download.
        SetApiKey(null);
        var freeArtifact = await _client.GetAsync($"/api/federated/runs/{createResponse.RunId}/artifact");
        Assert.Equal(HttpStatusCode.Forbidden, freeArtifact.StatusCode);

        // Option B (Pro) returns an encrypted artifact; key release is allowed without attestation.
        SetApiKey(proApiKey);
        var proArtifact = await _client.GetAsync($"/api/federated/runs/{createResponse.RunId}/artifact");
        proArtifact.EnsureSuccessStatusCode();
        Assert.Equal("true", proArtifact.Headers.GetValues("X-AiDotNet-Artifact-Encrypted").Single());
        var proEncryptedBytes = await proArtifact.Content.ReadAsByteArrayAsync();

        var proKeyContent = new StringContent("null", Encoding.UTF8, "application/json");
        var proKeyResponse = await _client.PostAsync($"/api/federated/runs/{createResponse.RunId}/artifact/key", proKeyContent);
        proKeyResponse.EnsureSuccessStatusCode();
        var proKey = await ReadAsJsonAsync<ModelArtifactKeyResponse>(proKeyResponse.Content);
        Assert.NotNull(proKey);

        var proBytes = DecryptAesGcmArtifact(createResponse.RunId, proEncryptedBytes, proKey!);

        var proPath = Path.Combine(Path.GetTempPath(), $"aidn-pro-{Guid.NewGuid():N}.model");
        await File.WriteAllBytesAsync(proPath, proBytes);
        _createdModelFiles.Add(proPath);

        var proModel = AiModelResult<double, Matrix<double>, Vector<double>>.LoadModel(
            proPath,
            metadata => new VectorModel<double>(new Vector<double>(metadata.FeatureCount > 0 ? metadata.FeatureCount : createResponse.ParameterCount)));

        var proParams = proModel.GetParameters();
        Assert.Equal(2.0, proParams[0], precision: 6);

        // Option C (Enterprise) returns encrypted artifact and requires key release via attestation.
        SetApiKey(enterpriseApiKey);
        var enterpriseArtifact = await _client.GetAsync($"/api/federated/runs/{createResponse.RunId}/artifact");
        enterpriseArtifact.EnsureSuccessStatusCode();
        Assert.Equal("true", enterpriseArtifact.Headers.GetValues("X-AiDotNet-Artifact-Encrypted").Single());

        var missingEvidenceContent = new StringContent("null", Encoding.UTF8, "application/json");
        var missingEvidence = await _client.PostAsync($"/api/federated/runs/{createResponse.RunId}/artifact/key", missingEvidenceContent);
        Assert.Equal(HttpStatusCode.BadRequest, missingEvidence.StatusCode);

        var keyResponse = await PostAsJsonAsync(
            $"/api/federated/runs/{createResponse.RunId}/artifact/key",
            CreateAttestationEvidence());
        keyResponse.EnsureSuccessStatusCode();
        var key = await ReadAsJsonAsync<ModelArtifactKeyResponse>(keyResponse.Content);
        Assert.NotNull(key);
        Assert.False(string.IsNullOrWhiteSpace(key!.KeyBase64));
    }

    [Fact]
    public async Task GetParameters_WhenClientNotJoined_Returns403()
    {
        var modelName = "fed-parameters-403";
        await LoadTrainedVectorModelAsync(modelName);

        SetApiKey(null);
        var create = await PostAsJsonAsync("/api/federated/runs", new CreateFederatedRunRequest
        {
            ModelName = modelName,
            Options = new FederatedLearningOptions(),
            MinClientUpdatesPerRound = 1
        });
        create.EnsureSuccessStatusCode();
        var createResponse = await ReadAsJsonAsync<CreateFederatedRunResponse>(create.Content);
        Assert.NotNull(createResponse);

        var response = await _client.GetAsync($"/api/federated/runs/{createResponse.RunId}/clients/12345/parameters");
        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
    }

    private void SetApiKey(string? apiKey)
    {
        _client.DefaultRequestHeaders.Remove(ApiKeyHeaderName);

        if (!string.IsNullOrWhiteSpace(apiKey))
        {
            _client.DefaultRequestHeaders.Add(ApiKeyHeaderName, apiKey);
        }
    }

    private async Task<string> CreateApiKeyAsync(SubscriptionTier tier)
    {
        using var scope = _factory.Services.CreateScope();
        var apiKeys = scope.ServiceProvider.GetRequiredService<IApiKeyService>();

        var created = await apiKeys.CreateAsync(new ApiKeyCreateRequest
        {
            Name = $"federated-tests-{tier}-{Guid.NewGuid():N}",
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

    private static AttestationEvidence CreateAttestationEvidence()
    {
        return new AttestationEvidence
        {
            Platform = "Windows",
            TeeType = "Development",
            Nonce = Guid.NewGuid().ToString("N"),
            AttestationToken = "dev"
        };
    }

    private async Task LoadTrainedVectorModelAsync(string modelName)
    {
        using var scope = _factory.Services.CreateScope();
        var options = scope.ServiceProvider.GetRequiredService<IOptions<ServingOptions>>().Value;

        var modelsRoot = Path.GetFullPath(options.ModelDirectory);
        Directory.CreateDirectory(modelsRoot);

        var fileName = $"{modelName}.model";
        var modelPath = Path.Combine(modelsRoot, fileName);

        await CreateAndSaveVectorModelArtifactAsync(modelPath);
        _createdModelFiles.Add(modelPath);

        var load = await PostAsJsonAsync("/api/models", new LoadModelRequest
        {
            Name = modelName,
            Path = fileName,
            NumericType = NumericType.Double
        });

        if (!load.IsSuccessStatusCode)
        {
            var body = await load.Content.ReadAsStringAsync();
            throw new InvalidOperationException(
                $"Failed to load model '{modelName}' from '{fileName}' (HTTP {(int)load.StatusCode}). Body: {body}");
        }
        _loadedModels.Add(modelName);
    }

    private static async Task CreateAndSaveVectorModelArtifactAsync(string modelPath)
    {
        const int rows = 30;
        const int cols = 3;

        var features = new double[rows, cols];
        var labels = new double[rows];

        for (int i = 0; i < rows; i++)
        {
            double a = i + 1;
            double b = i + 2;
            double c = i + 3;
            features[i, 0] = a;
            features[i, 1] = b;
            features[i, 2] = c;
            labels[i] = a + b + c;
        }

        var dataLoader = DataLoaders.FromArrays(features, labels);

        var model = new VectorModel<double>(new Vector<double>(new[] { 0.0, 0.0, 0.0 }));
        var optimizer = new SinglePassTestOptimizer(model);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(dataLoader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .BuildAsync();

        result.SaveModel(modelPath);
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

    private void CleanupCreatedFiles()
    {
        foreach (var path in _createdModelFiles)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch
            {
                // Best-effort cleanup for test artifacts.
            }
        }

        _createdModelFiles.Clear();
    }
}
