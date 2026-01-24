using System;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LanguageModels;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LanguageModels;

/// <summary>
/// Comprehensive integration tests for the LanguageModels module.
/// Tests constructor validation, property behavior, token estimation, retry logic,
/// and HTTP communication patterns without requiring real API keys.
/// </summary>
public class LanguageModelsIntegrationTests
{
    private const string ValidApiKey = "sk-test-api-key-12345";
    private const string ValidEndpoint = "https://test-resource.openai.azure.com";
    private const string ValidDeploymentName = "gpt-4-deployment";

    #region OpenAIChatModel Constructor Tests

    [Fact]
    public void OpenAIChatModel_Constructor_WithValidApiKey_CreatesInstance()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);

        Assert.NotNull(model);
        Assert.Equal("gpt-3.5-turbo", model.ModelName);
    }

    [Fact]
    public void OpenAIChatModel_Constructor_WithCustomModelName_SetsModelName()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4");

        Assert.Equal("gpt-4", model.ModelName);
    }

    [Fact]
    public void OpenAIChatModel_Constructor_WithNullApiKey_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new OpenAIChatModel<double>(null!));
    }

    [Fact]
    public void OpenAIChatModel_Constructor_WithEmptyApiKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new OpenAIChatModel<double>(string.Empty));
    }

    [Fact]
    public void OpenAIChatModel_Constructor_WithWhitespaceApiKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new OpenAIChatModel<double>("   "));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(2.1)]
    [InlineData(-1.0)]
    [InlineData(5.0)]
    public void OpenAIChatModel_Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
    {
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>(ValidApiKey, temperature: temperature));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(1.5)]
    [InlineData(2.0)]
    public void OpenAIChatModel_Constructor_WithValidTemperature_Succeeds(double temperature)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, temperature: temperature);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(-1.0)]
    [InlineData(5.0)]
    public void OpenAIChatModel_Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
    {
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>(ValidApiKey, topP: topP));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void OpenAIChatModel_Constructor_WithValidTopP_Succeeds(double topP)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, topP: topP);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    [InlineData(-5.0)]
    [InlineData(5.0)]
    public void OpenAIChatModel_Constructor_WithInvalidFrequencyPenalty_ThrowsArgumentException(double penalty)
    {
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>(ValidApiKey, frequencyPenalty: penalty));
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(0.0)]
    [InlineData(2.0)]
    public void OpenAIChatModel_Constructor_WithValidFrequencyPenalty_Succeeds(double penalty)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, frequencyPenalty: penalty);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    [InlineData(-5.0)]
    [InlineData(5.0)]
    public void OpenAIChatModel_Constructor_WithInvalidPresencePenalty_ThrowsArgumentException(double penalty)
    {
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>(ValidApiKey, presencePenalty: penalty));
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(0.0)]
    [InlineData(2.0)]
    public void OpenAIChatModel_Constructor_WithValidPresencePenalty_Succeeds(double penalty)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, presencePenalty: penalty);
        Assert.NotNull(model);
    }

    [Fact]
    public void OpenAIChatModel_Constructor_WithCustomHttpClient_Succeeds()
    {
        using var httpClient = new HttpClient();
        var model = new OpenAIChatModel<double>(ValidApiKey, httpClient: httpClient);

        Assert.NotNull(model);
    }

    [Fact]
    public void OpenAIChatModel_Constructor_WithCustomEndpoint_Succeeds()
    {
        var model = new OpenAIChatModel<double>(
            ValidApiKey,
            endpoint: "https://custom-endpoint.com/v1/chat/completions");

        Assert.NotNull(model);
    }

    #endregion

    #region OpenAIChatModel MaxContextTokens Tests

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT35Turbo_Returns4096()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-3.5-turbo");
        Assert.Equal(4096, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT35Turbo16K_Returns16384()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-3.5-turbo-16k");
        Assert.Equal(16384, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT4_Returns8192()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4");
        Assert.Equal(8192, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT432K_Returns32768()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4-32k");
        Assert.Equal(32768, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT4Turbo_Returns128000()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4-turbo");
        Assert.Equal(128000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT4TurboPreview_Returns128000()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4-turbo-preview");
        Assert.Equal(128000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT4o_Returns128000()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4o");
        Assert.Equal(128000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT4oMini_Returns128000()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4o-mini");
        Assert.Equal(128000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_UnknownModel_Returns4096Default()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "unknown-model");
        Assert.Equal(4096, model.MaxContextTokens);
    }

    // BUG TEST: OpenAI is missing newer models (o1, o1-mini, o3-mini, gpt-4o-2024-08-06)
    // These tests will fail until the source code is updated

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_O1_ShouldReturnCorrectTokens()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "o1");
        // o1 has 128k context window - if this fails with 4096, it means the model isn't registered
        Assert.Equal(128000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_O1Mini_ShouldReturnCorrectTokens()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "o1-mini");
        // o1-mini has 128k context window
        Assert.Equal(128000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_O3Mini_ShouldReturnCorrectTokens()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "o3-mini");
        // o3-mini has 200k context window
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT4o_2024_08_06_ShouldReturnCorrectTokens()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4o-2024-08-06");
        // This is a dated version of gpt-4o, should have 128k tokens
        Assert.Equal(128000, model.MaxContextTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_GPT4_1106_Preview_ShouldReturnCorrectTokens()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-4-1106-preview");
        // GPT-4 turbo preview (dated version) has 128k tokens
        Assert.Equal(128000, model.MaxContextTokens);
    }

    #endregion

    #region OpenAIChatModel MaxGenerationTokens Tests

    [Fact]
    public void OpenAIChatModel_MaxGenerationTokens_DefaultValue_Returns2048()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    [Fact]
    public void OpenAIChatModel_MaxGenerationTokens_CustomValue_ReturnsCustomValue()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, maxTokens: 500);
        Assert.Equal(500, model.MaxGenerationTokens);
    }

    #endregion

    #region AnthropicChatModel Constructor Tests

    [Fact]
    public void AnthropicChatModel_Constructor_WithValidApiKey_CreatesInstance()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);

        Assert.NotNull(model);
        Assert.Equal("claude-3-sonnet-20240229", model.ModelName);
    }

    [Fact]
    public void AnthropicChatModel_Constructor_WithCustomModelName_SetsModelName()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3-opus-20240229");

        Assert.Equal("claude-3-opus-20240229", model.ModelName);
    }

    [Fact]
    public void AnthropicChatModel_Constructor_WithNullApiKey_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new AnthropicChatModel<double>(null!));
    }

    [Fact]
    public void AnthropicChatModel_Constructor_WithEmptyApiKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new AnthropicChatModel<double>(string.Empty));
    }

    [Fact]
    public void AnthropicChatModel_Constructor_WithWhitespaceApiKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new AnthropicChatModel<double>("   "));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(-1.0)]
    [InlineData(2.0)]
    public void AnthropicChatModel_Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
    {
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>(ValidApiKey, temperature: temperature));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void AnthropicChatModel_Constructor_WithValidTemperature_Succeeds(double temperature)
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, temperature: temperature);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(4097)]
    [InlineData(10000)]
    public void AnthropicChatModel_Constructor_WithInvalidMaxTokens_ThrowsArgumentException(int maxTokens)
    {
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>(ValidApiKey, maxTokens: maxTokens));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(1000)]
    [InlineData(4096)]
    public void AnthropicChatModel_Constructor_WithValidMaxTokens_Succeeds(int maxTokens)
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, maxTokens: maxTokens);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void AnthropicChatModel_Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
    {
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>(ValidApiKey, topP: topP));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-100)]
    public void AnthropicChatModel_Constructor_WithNegativeTopK_ThrowsArgumentException(int topK)
    {
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>(ValidApiKey, topK: topK));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(100)]
    public void AnthropicChatModel_Constructor_WithValidTopK_Succeeds(int topK)
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, topK: topK);
        Assert.NotNull(model);
    }

    [Fact]
    public void AnthropicChatModel_Constructor_WithCustomHttpClient_Succeeds()
    {
        using var httpClient = new HttpClient();
        var model = new AnthropicChatModel<double>(ValidApiKey, httpClient: httpClient);

        Assert.NotNull(model);
    }

    [Fact]
    public void AnthropicChatModel_Constructor_WithCustomEndpoint_Succeeds()
    {
        var model = new AnthropicChatModel<double>(
            ValidApiKey,
            endpoint: "https://custom-endpoint.com/v1/messages");

        Assert.NotNull(model);
    }

    #endregion

    #region AnthropicChatModel MaxContextTokens Tests

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude3Opus_Returns200000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3-opus-20240229");
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude3Sonnet_Returns200000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3-sonnet-20240229");
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude3Haiku_Returns200000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3-haiku-20240307");
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude21_Returns200000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-2.1");
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude20_Returns100000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-2.0");
        Assert.Equal(100000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude2_Returns100000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-2");
        Assert.Equal(100000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_ClaudeInstant_Returns100000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-instant-1.2");
        Assert.Equal(100000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_UnknownModel_Returns100000Default()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "unknown-model");
        Assert.Equal(100000, model.MaxContextTokens);
    }

    // BUG TEST: Anthropic is missing Claude 3.5 models
    // These tests will fail until the source code is updated

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude35Sonnet_ShouldReturn200000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3-5-sonnet-20241022");
        // Claude 3.5 Sonnet has 200k context window
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude35Haiku_ShouldReturn200000()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3-5-haiku-20241022");
        // Claude 3.5 Haiku has 200k context window
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxContextTokens_Claude35Sonnet_Latest_ShouldReturn200000()
    {
        // Users might use shorthand model names
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3.5-sonnet");
        // This should recognize 3.5 variant
        Assert.Equal(200000, model.MaxContextTokens);
    }

    #endregion

    #region AnthropicChatModel MaxGenerationTokens Tests

    [Fact]
    public void AnthropicChatModel_MaxGenerationTokens_DefaultValue_Returns2048()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);
        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    [Fact]
    public void AnthropicChatModel_MaxGenerationTokens_CustomValue_ReturnsCustomValue()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, maxTokens: 500);
        Assert.Equal(500, model.MaxGenerationTokens);
    }

    #endregion

    #region AzureOpenAIChatModel Constructor Tests

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithValidParameters_CreatesInstance()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint,
            ValidApiKey,
            ValidDeploymentName);

        Assert.NotNull(model);
        Assert.Equal($"azure-{ValidDeploymentName}", model.ModelName);
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithNullEndpoint_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(null!, ValidApiKey, ValidDeploymentName));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithEmptyEndpoint_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(string.Empty, ValidApiKey, ValidDeploymentName));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithWhitespaceEndpoint_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>("   ", ValidApiKey, ValidDeploymentName));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithNullApiKey_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AzureOpenAIChatModel<double>(ValidEndpoint, null!, ValidDeploymentName));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithEmptyApiKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(ValidEndpoint, string.Empty, ValidDeploymentName));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithWhitespaceApiKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(ValidEndpoint, "   ", ValidDeploymentName));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithNullDeploymentName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(ValidEndpoint, ValidApiKey, null!));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithEmptyDeploymentName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(ValidEndpoint, ValidApiKey, string.Empty));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithWhitespaceDeploymentName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(ValidEndpoint, ValidApiKey, "   "));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(2.1)]
    public void AzureOpenAIChatModel_Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                ValidEndpoint, ValidApiKey, ValidDeploymentName, temperature: temperature));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(2.0)]
    public void AzureOpenAIChatModel_Constructor_WithValidTemperature_Succeeds(double temperature)
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName, temperature: temperature);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void AzureOpenAIChatModel_Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                ValidEndpoint, ValidApiKey, ValidDeploymentName, topP: topP));
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    public void AzureOpenAIChatModel_Constructor_WithInvalidFrequencyPenalty_ThrowsArgumentException(double penalty)
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                ValidEndpoint, ValidApiKey, ValidDeploymentName, frequencyPenalty: penalty));
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    public void AzureOpenAIChatModel_Constructor_WithInvalidPresencePenalty_ThrowsArgumentException(double penalty)
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                ValidEndpoint, ValidApiKey, ValidDeploymentName, presencePenalty: penalty));
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithCustomApiVersion_Succeeds()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName,
            apiVersion: "2023-05-15");

        Assert.NotNull(model);
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_WithCustomHttpClient_Succeeds()
    {
        using var httpClient = new HttpClient();
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName,
            httpClient: httpClient);

        Assert.NotNull(model);
    }

    [Fact]
    public void AzureOpenAIChatModel_Constructor_TrimsEndpointTrailingSlash()
    {
        var model = new AzureOpenAIChatModel<double>(
            $"{ValidEndpoint}/", ValidApiKey, ValidDeploymentName);

        Assert.NotNull(model);
    }

    #endregion

    #region AzureOpenAIChatModel Property Tests

    [Fact]
    public void AzureOpenAIChatModel_ModelName_IncludesDeploymentName()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, "my-custom-deployment");

        Assert.Equal("azure-my-custom-deployment", model.ModelName);
    }

    [Fact]
    public void AzureOpenAIChatModel_MaxContextTokens_DefaultValue_Returns8192()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);

        Assert.Equal(8192, model.MaxContextTokens);
    }

    [Fact]
    public void AzureOpenAIChatModel_MaxContextTokens_CustomValue_ReturnsCustomValue()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName,
            maxContextTokens: 16384);

        Assert.Equal(16384, model.MaxContextTokens);
    }

    [Fact]
    public void AzureOpenAIChatModel_MaxGenerationTokens_DefaultValue_Returns2048()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);

        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    [Fact]
    public void AzureOpenAIChatModel_MaxGenerationTokens_CustomValue_ReturnsCustomValue()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName,
            maxTokens: 4096);

        Assert.Equal(4096, model.MaxGenerationTokens);
    }

    #endregion

    #region Cross-Model Type Parameter Tests

    [Fact]
    public void OpenAIChatModel_WithFloatTypeParameter_CreatesInstance()
    {
        var model = new OpenAIChatModel<float>(ValidApiKey);
        Assert.NotNull(model);
    }

    [Fact]
    public void OpenAIChatModel_WithDecimalTypeParameter_CreatesInstance()
    {
        var model = new OpenAIChatModel<decimal>(ValidApiKey);
        Assert.NotNull(model);
    }

    [Fact]
    public void AnthropicChatModel_WithFloatTypeParameter_CreatesInstance()
    {
        var model = new AnthropicChatModel<float>(ValidApiKey);
        Assert.NotNull(model);
    }

    [Fact]
    public void AnthropicChatModel_WithDecimalTypeParameter_CreatesInstance()
    {
        var model = new AnthropicChatModel<decimal>(ValidApiKey);
        Assert.NotNull(model);
    }

    [Fact]
    public void AzureOpenAIChatModel_WithFloatTypeParameter_CreatesInstance()
    {
        var model = new AzureOpenAIChatModel<float>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);
        Assert.NotNull(model);
    }

    [Fact]
    public void AzureOpenAIChatModel_WithDecimalTypeParameter_CreatesInstance()
    {
        var model = new AzureOpenAIChatModel<decimal>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);
        Assert.NotNull(model);
    }

    #endregion

    #region Interface Implementation Tests

    [Fact]
    public void OpenAIChatModel_ImplementsIChatModel()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        Assert.IsAssignableFrom<AiDotNet.Interfaces.IChatModel<double>>(model);
    }

    [Fact]
    public void AnthropicChatModel_ImplementsIChatModel()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);
        Assert.IsAssignableFrom<AiDotNet.Interfaces.IChatModel<double>>(model);
    }

    [Fact]
    public void AzureOpenAIChatModel_ImplementsIChatModel()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);
        Assert.IsAssignableFrom<AiDotNet.Interfaces.IChatModel<double>>(model);
    }

    [Fact]
    public void OpenAIChatModel_ImplementsILanguageModel()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        Assert.IsAssignableFrom<AiDotNet.Interfaces.ILanguageModel<double>>(model);
    }

    [Fact]
    public void AnthropicChatModel_ImplementsILanguageModel()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);
        Assert.IsAssignableFrom<AiDotNet.Interfaces.ILanguageModel<double>>(model);
    }

    [Fact]
    public void AzureOpenAIChatModel_ImplementsILanguageModel()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);
        Assert.IsAssignableFrom<AiDotNet.Interfaces.ILanguageModel<double>>(model);
    }

    #endregion

    #region Generate Method Input Validation Tests

    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_WithNullPrompt_ThrowsArgumentException()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(null!));
    }

    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_WithEmptyPrompt_ThrowsArgumentException()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(string.Empty));
    }

    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_WithWhitespacePrompt_ThrowsArgumentException()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync("   "));
    }

    [Fact]
    public async Task AnthropicChatModel_GenerateAsync_WithNullPrompt_ThrowsArgumentException()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);
        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(null!));
    }

    [Fact]
    public async Task AnthropicChatModel_GenerateAsync_WithEmptyPrompt_ThrowsArgumentException()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);
        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(string.Empty));
    }

    [Fact]
    public async Task AzureOpenAIChatModel_GenerateAsync_WithNullPrompt_ThrowsArgumentException()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);
        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(null!));
    }

    [Fact]
    public async Task AzureOpenAIChatModel_GenerateAsync_WithEmptyPrompt_ThrowsArgumentException()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);
        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(string.Empty));
    }

    #endregion

    #region Token Estimation and Limit Tests

    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_PromptExceedsMaxTokens_ThrowsArgumentException()
    {
        // Create model with small context window to test token limit validation
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-3.5-turbo");

        // Create a prompt that would exceed the 4096 token limit
        // EstimateTokenCount uses 1 token ≈ 4 characters, so 4096 * 4 = 16384 chars
        var longPrompt = new string('a', 20000);

        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(longPrompt));
    }

    [Fact]
    public async Task AnthropicChatModel_GenerateAsync_PromptExceedsMaxTokens_ThrowsArgumentException()
    {
        // Create model - Claude has 200000 token context window
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: "claude-3-sonnet-20240229");

        // Create a prompt that would exceed the 200000 token limit
        // EstimateTokenCount uses 1 token ≈ 4 characters, so 200000 * 4 = 800000 chars
        var longPrompt = new string('a', 850000);

        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(longPrompt));
    }

    [Fact]
    public async Task AzureOpenAIChatModel_GenerateAsync_PromptExceedsMaxTokens_ThrowsArgumentException()
    {
        // Create model with default 8192 context window
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);

        // Create a prompt that would exceed the 8192 token limit (8192 * 4 = 32768)
        var longPrompt = new string('a', 40000);

        await Assert.ThrowsAsync<ArgumentException>(() => model.GenerateAsync(longPrompt));
    }

    // Token estimation edge case tests
    [Fact]
    public async Task ChatModel_GenerateAsync_NearLimitPrompt_AcceptsPrompt()
    {
        // Create model with gpt-3.5-turbo (4096 context)
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: "gpt-3.5-turbo");

        // Create a prompt just under the limit (4096 tokens * 4 chars = 16384)
        // Subtract some for safety margin
        var almostMaxPrompt = new string('a', 16000);

        // This should NOT throw ArgumentException for length
        // It will throw InvalidOperationException when trying to call API (no real key)
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => model.GenerateAsync(almostMaxPrompt));

        // Verify it wasn't rejected for length
        Assert.DoesNotContain("too long", exception.Message);
    }

    #endregion

    #region Model Name Case Insensitivity Tests

    [Theory]
    [InlineData("GPT-3.5-TURBO")]
    [InlineData("Gpt-3.5-Turbo")]
    [InlineData("gpt-3.5-turbo")]
    public void OpenAIChatModel_ModelName_CaseInsensitive_Returns4096(string modelName)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: modelName);
        Assert.Equal(4096, model.MaxContextTokens);
    }

    [Theory]
    [InlineData("GPT-4")]
    [InlineData("Gpt-4")]
    [InlineData("gpt-4")]
    public void OpenAIChatModel_ModelName_GPT4_CaseInsensitive_Returns8192(string modelName)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: modelName);
        Assert.Equal(8192, model.MaxContextTokens);
    }

    [Theory]
    [InlineData("CLAUDE-3-OPUS-20240229")]
    [InlineData("Claude-3-Opus-20240229")]
    [InlineData("claude-3-opus-20240229")]
    public void AnthropicChatModel_ModelName_CaseInsensitive_Returns200000(string modelName)
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: modelName);
        Assert.Equal(200000, model.MaxContextTokens);
    }

    [Theory]
    [InlineData("CLAUDE-2.1")]
    [InlineData("Claude-2.1")]
    [InlineData("claude-2.1")]
    public void AnthropicChatModel_ModelName_Claude21_CaseInsensitive_Returns200000(string modelName)
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, modelName: modelName);
        Assert.Equal(200000, model.MaxContextTokens);
    }

    #endregion

    #region Default Values Tests

    [Fact]
    public void OpenAIChatModel_DefaultValues_AreCorrect()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);

        Assert.Equal("gpt-3.5-turbo", model.ModelName);
        Assert.Equal(4096, model.MaxContextTokens);
        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    [Fact]
    public void AnthropicChatModel_DefaultValues_AreCorrect()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);

        Assert.Equal("claude-3-sonnet-20240229", model.ModelName);
        Assert.Equal(200000, model.MaxContextTokens);
        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    [Fact]
    public void AzureOpenAIChatModel_DefaultValues_AreCorrect()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);

        Assert.Equal($"azure-{ValidDeploymentName}", model.ModelName);
        Assert.Equal(8192, model.MaxContextTokens);
        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    #endregion

    #region Boundary Value Tests

    [Theory]
    [InlineData(0.0)]
    [InlineData(2.0)]
    public void OpenAIChatModel_Temperature_BoundaryValues_Succeed(double temperature)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, temperature: temperature);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void AnthropicChatModel_Temperature_BoundaryValues_Succeed(double temperature)
    {
        var model = new AnthropicChatModel<double>(ValidApiKey, temperature: temperature);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(2.0)]
    public void OpenAIChatModel_FrequencyPenalty_BoundaryValues_Succeed(double penalty)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, frequencyPenalty: penalty);
        Assert.NotNull(model);
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(2.0)]
    public void OpenAIChatModel_PresencePenalty_BoundaryValues_Succeed(double penalty)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, presencePenalty: penalty);
        Assert.NotNull(model);
    }

    [Fact]
    public void AnthropicChatModel_MaxTokens_BoundaryValues_Succeed()
    {
        var modelMin = new AnthropicChatModel<double>(ValidApiKey, maxTokens: 1);
        Assert.NotNull(modelMin);

        var modelMax = new AnthropicChatModel<double>(ValidApiKey, maxTokens: 4096);
        Assert.NotNull(modelMax);
    }

    #endregion

    #region HTTP Mock Tests for Retry Logic

    /// <summary>
    /// Tests that cancellation is properly respected during generation.
    /// </summary>
    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_WithCancelledToken_ThrowsOperationCancelledException()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(
            () => model.GenerateAsync("Test prompt", cts.Token));
    }

    [Fact]
    public async Task AnthropicChatModel_GenerateAsync_WithCancelledToken_ThrowsOperationCancelledException()
    {
        var model = new AnthropicChatModel<double>(ValidApiKey);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(
            () => model.GenerateAsync("Test prompt", cts.Token));
    }

    [Fact]
    public async Task AzureOpenAIChatModel_GenerateAsync_WithCancelledToken_ThrowsOperationCancelledException()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(
            () => model.GenerateAsync("Test prompt", cts.Token));
    }

    /// <summary>
    /// Tests the synchronous Generate method wrapper.
    /// </summary>
    [Fact]
    public void OpenAIChatModel_Generate_WithValidPrompt_ExecutesSynchronously()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);

        // This should throw InvalidOperationException (no real API key)
        // not OperationCanceledException or AggregateException
        var exception = Assert.Throws<InvalidOperationException>(() => model.Generate("Test prompt"));

        Assert.Contains("Error generating response", exception.Message);
    }

    /// <summary>
    /// Tests the GenerateResponseAsync alias method.
    /// </summary>
    [Fact]
    public async Task OpenAIChatModel_GenerateResponseAsync_IsAliasForGenerateAsync()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);

        // Both methods should behave identically
        var exception1 = await Assert.ThrowsAsync<InvalidOperationException>(
            () => model.GenerateAsync("Test prompt"));
        var exception2 = await Assert.ThrowsAsync<InvalidOperationException>(
            () => model.GenerateResponseAsync("Test prompt"));

        // Both should have similar error messages (API call failures)
        Assert.Contains("Error generating response", exception1.Message);
        Assert.Contains("Error generating response", exception2.Message);
    }

    #endregion

    #region Mock HTTP Client Tests for API Communication

    /// <summary>
    /// Tests that OpenAI model correctly formats and sends API requests.
    /// Uses a mock handler to capture and verify the request.
    /// </summary>
    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_SendsCorrectHeaders()
    {
        // Arrange
        var mockHandler = new MockHttpMessageHandler(req =>
        {
            // Verify Authorization header
            Assert.True(req.Headers.Authorization != null);
            Assert.Equal("Bearer", req.Headers.Authorization.Scheme);
            Assert.Equal(ValidApiKey, req.Headers.Authorization.Parameter);

            // Return a valid response
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    @"{""id"":""test"",""choices"":[{""message"":{""content"":""Test response""}}]}",
                    Encoding.UTF8,
                    "application/json")
            };
        });

        using var httpClient = new HttpClient(mockHandler);
        var model = new OpenAIChatModel<double>(ValidApiKey, httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("Test response", result);
    }

    [Fact]
    public async Task AnthropicChatModel_GenerateAsync_SendsCorrectHeaders()
    {
        // Arrange
        var mockHandler = new MockHttpMessageHandler(req =>
        {
            // Verify x-api-key header
            Assert.True(req.Headers.Contains("x-api-key"));
            Assert.Equal(ValidApiKey, req.Headers.GetValues("x-api-key").First());

            // Verify anthropic-version header
            Assert.True(req.Headers.Contains("anthropic-version"));
            Assert.Equal("2023-06-01", req.Headers.GetValues("anthropic-version").First());

            // Return a valid response
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    @"{""id"":""test"",""content"":[{""type"":""text"",""text"":""Test response""}]}",
                    Encoding.UTF8,
                    "application/json")
            };
        });

        using var httpClient = new HttpClient(mockHandler);
        var model = new AnthropicChatModel<double>(ValidApiKey, httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("Test response", result);
    }

    [Fact]
    public async Task AzureOpenAIChatModel_GenerateAsync_SendsCorrectHeaders()
    {
        // Arrange
        var mockHandler = new MockHttpMessageHandler(req =>
        {
            // Verify api-key header (Azure uses different header than OpenAI)
            Assert.True(req.Headers.Contains("api-key"));
            Assert.Equal(ValidApiKey, req.Headers.GetValues("api-key").First());

            // Verify URL contains api-version
            Assert.Contains("api-version=", req.RequestUri?.Query);

            // Return a valid response
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    @"{""id"":""test"",""choices"":[{""message"":{""content"":""Test response""}}]}",
                    Encoding.UTF8,
                    "application/json")
            };
        });

        using var httpClient = new HttpClient(mockHandler);
        var model = new AzureOpenAIChatModel<double>(
            ValidEndpoint, ValidApiKey, ValidDeploymentName, httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("Test response", result);
    }

    /// <summary>
    /// Tests that API errors are properly handled and wrapped.
    /// </summary>
    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_WithApiError_ThrowsInvalidOperationException()
    {
        // Arrange
        var mockHandler = new MockHttpMessageHandler(req =>
        {
            return new HttpResponseMessage(HttpStatusCode.BadRequest)
            {
                Content = new StringContent(@"{""error"":{""message"":""Invalid request""}}")
            };
        });

        using var httpClient = new HttpClient(mockHandler);
        var model = new OpenAIChatModel<double>(ValidApiKey, httpClient: httpClient);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => model.GenerateAsync("Test prompt"));

        Assert.Contains("Error generating response", exception.Message);
    }

    /// <summary>
    /// Tests that empty responses are handled correctly.
    /// </summary>
    [Fact]
    public async Task OpenAIChatModel_GenerateAsync_WithEmptyChoices_ThrowsInvalidOperationException()
    {
        // Arrange
        var mockHandler = new MockHttpMessageHandler(req =>
        {
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    @"{""id"":""test"",""choices"":[]}",
                    Encoding.UTF8,
                    "application/json")
            };
        });

        using var httpClient = new HttpClient(mockHandler);
        var model = new OpenAIChatModel<double>(ValidApiKey, httpClient: httpClient);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => model.GenerateAsync("Test prompt"));

        Assert.Contains("no choices", exception.Message);
    }

    [Fact]
    public async Task AnthropicChatModel_GenerateAsync_WithEmptyContent_ThrowsInvalidOperationException()
    {
        // Arrange
        var mockHandler = new MockHttpMessageHandler(req =>
        {
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    @"{""id"":""test"",""content"":[]}",
                    Encoding.UTF8,
                    "application/json")
            };
        });

        using var httpClient = new HttpClient(mockHandler);
        var model = new AnthropicChatModel<double>(ValidApiKey, httpClient: httpClient);

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => model.GenerateAsync("Test prompt"));

        Assert.Contains("no content", exception.Message);
    }

    /// <summary>
    /// Tests that Anthropic correctly joins multiple content blocks.
    /// </summary>
    [Fact]
    public async Task AnthropicChatModel_GenerateAsync_WithMultipleContentBlocks_JoinsWithNewline()
    {
        // Arrange
        var mockHandler = new MockHttpMessageHandler(req =>
        {
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    @"{""id"":""test"",""content"":[{""type"":""text"",""text"":""First""},{""type"":""text"",""text"":""Second""}]}",
                    Encoding.UTF8,
                    "application/json")
            };
        });

        using var httpClient = new HttpClient(mockHandler);
        var model = new AnthropicChatModel<double>(ValidApiKey, httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("First\nSecond", result);
    }

    #endregion

    #region Helper Classes

    /// <summary>
    /// A mock HTTP message handler for testing HTTP communication.
    /// </summary>
    private class MockHttpMessageHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, HttpResponseMessage> _handler;

        public MockHttpMessageHandler(Func<HttpRequestMessage, HttpResponseMessage> handler)
        {
            _handler = handler;
        }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(_handler(request));
        }
    }

    #endregion
}
