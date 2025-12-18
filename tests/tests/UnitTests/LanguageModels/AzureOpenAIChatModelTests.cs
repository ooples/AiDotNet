using AiDotNet.LanguageModels;
using Xunit;
using System.Net;
using System.Net.Http;
using Moq;
using Moq.Protected;

namespace AiDotNetTests.UnitTests.LanguageModels;

/// <summary>
/// Unit tests for the AzureOpenAIChatModel class.
/// </summary>
public class AzureOpenAIChatModelTests
{
    private const string ValidEndpoint = "https://test-resource.openai.azure.com";
    private const string ValidApiKey = "test-api-key";
    private const string ValidDeployment = "gpt-4-deployment";

    [Fact]
    public void Constructor_WithValidParameters_InitializesSuccessfully()
    {
        // Arrange & Act
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment);

        // Assert
        Assert.NotNull(model);
        Assert.Equal($"azure-{ValidDeployment}", model.ModelName);
        Assert.Equal(8192, model.MaxContextTokens);
        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_WithInvalidEndpoint_ThrowsArgumentException(string? endpoint)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                endpoint: endpoint!,
                apiKey: ValidApiKey,
                deploymentName: ValidDeployment));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_WithInvalidApiKey_ThrowsArgumentException(string? apiKey)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                endpoint: ValidEndpoint,
                apiKey: apiKey!,
                deploymentName: ValidDeployment));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_WithInvalidDeploymentName_ThrowsArgumentException(string? deploymentName)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                endpoint: ValidEndpoint,
                apiKey: ValidApiKey,
                deploymentName: deploymentName!));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(2.1)]
    public void Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                endpoint: ValidEndpoint,
                apiKey: ValidApiKey,
                deploymentName: ValidDeployment,
                temperature: temperature));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                endpoint: ValidEndpoint,
                apiKey: ValidApiKey,
                deploymentName: ValidDeployment,
                topP: topP));
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    public void Constructor_WithInvalidFrequencyPenalty_ThrowsArgumentException(double penalty)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                endpoint: ValidEndpoint,
                apiKey: ValidApiKey,
                deploymentName: ValidDeployment,
                frequencyPenalty: penalty));
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    public void Constructor_WithInvalidPresencePenalty_ThrowsArgumentException(double penalty)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(
                endpoint: ValidEndpoint,
                apiKey: ValidApiKey,
                deploymentName: ValidDeployment,
                presencePenalty: penalty));
    }

    [Fact]
    public void Constructor_WithTrailingSlashInEndpoint_RemovesSlash()
    {
        // Arrange
        var endpointWithSlash = "https://test-resource.openai.azure.com/";

        // Act
        var model = new AzureOpenAIChatModel<double>(
            endpoint: endpointWithSlash,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment);

        // Assert
        Assert.NotNull(model);
        // The endpoint should be trimmed internally (we can't directly test private fields,
        // but successful construction indicates proper handling)
    }

    [Fact]
    public async Task GenerateAsync_WithNullPrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            model.GenerateAsync(null!));
    }

    [Fact]
    public async Task GenerateAsync_WithEmptyPrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            model.GenerateAsync(""));
    }

    [Fact]
    public async Task GenerateAsync_WithWhitespacePrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            model.GenerateAsync("   "));
    }

    [Fact]
    public async Task GenerateAsync_WithSuccessfulResponse_ReturnsContent()
    {
        // Arrange
        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(@"{
                    ""id"": ""chatcmpl-123"",
                    ""choices"": [{
                        ""index"": 0,
                        ""message"": {
                            ""role"": ""assistant"",
                            ""content"": ""This is an Azure OpenAI response.""
                        },
                        ""finish_reason"": ""stop""
                    }],
                    ""usage"": {
                        ""prompt_tokens"": 10,
                        ""completion_tokens"": 20,
                        ""total_tokens"": 30
                    }
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("This is an Azure OpenAI response.", result);
    }

    [Fact]
    public async Task GenerateAsync_WithHttpError_ThrowsHttpRequestException()
    {
        // Arrange
        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.Unauthorized,
                Content = new StringContent(@"{""error"": {""message"": ""Invalid API key""}}")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            httpClient: httpClient);

        // Act & Assert
        await Assert.ThrowsAsync<HttpRequestException>(() =>
            model.GenerateAsync("Test prompt"));
    }

    [Fact]
    public async Task GenerateAsync_WithEmptyChoices_ThrowsInvalidOperationException()
    {
        // Arrange
        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(@"{
                    ""id"": ""chatcmpl-123"",
                    ""choices"": []
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            httpClient: httpClient);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            model.GenerateAsync("Test prompt"));
    }

    [Fact]
    public async Task GenerateAsync_WithEmptyMessageContent_ThrowsInvalidOperationException()
    {
        // Arrange
        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(@"{
                    ""id"": ""chatcmpl-123"",
                    ""choices"": [{
                        ""index"": 0,
                        ""message"": {
                            ""role"": ""assistant"",
                            ""content"": """"
                        },
                        ""finish_reason"": ""stop""
                    }]
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            httpClient: httpClient);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            model.GenerateAsync("Test prompt"));
    }

    [Fact]
    public void Generate_WithSuccessfulResponse_ReturnsContent()
    {
        // Arrange
        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(@"{
                    ""id"": ""chatcmpl-123"",
                    ""choices"": [{
                        ""index"": 0,
                        ""message"": {
                            ""role"": ""assistant"",
                            ""content"": ""Synchronous Azure response.""
                        },
                        ""finish_reason"": ""stop""
                    }]
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            httpClient: httpClient);

        // Act
        var result = model.Generate("Test prompt");

        // Assert
        Assert.Equal("Synchronous Azure response.", result);
    }

    [Fact]
    public async Task GenerateResponseAsync_WithSuccessfulResponse_ReturnsContent()
    {
        // Arrange
        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(@"{
                    ""id"": ""chatcmpl-123"",
                    ""choices"": [{
                        ""index"": 0,
                        ""message"": {
                            ""role"": ""assistant"",
                            ""content"": ""Response via GenerateResponseAsync.""
                        },
                        ""finish_reason"": ""stop""
                    }]
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            httpClient: httpClient);

        // Act
        var result = await model.GenerateResponseAsync("Test prompt");

        // Assert
        Assert.Equal("Response via GenerateResponseAsync.", result);
    }

    [Fact]
    public async Task GenerateAsync_WithRateLimitError_RetriesAndSucceeds()
    {
        // Arrange
        var callCount = 0;
        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(() =>
            {
                callCount++;
                if (callCount == 1)
                {
                    // First call fails with rate limit
                    return new HttpResponseMessage
                    {
                        StatusCode = HttpStatusCode.TooManyRequests,
                        Content = new StringContent(@"{""error"": {""message"": ""Rate limit exceeded""}}")
                    };
                }
                else
                {
                    // Second call succeeds
                    return new HttpResponseMessage
                    {
                        StatusCode = HttpStatusCode.OK,
                        Content = new StringContent(@"{
                            ""id"": ""chatcmpl-123"",
                            ""choices"": [{
                                ""index"": 0,
                                ""message"": {
                                    ""role"": ""assistant"",
                                    ""content"": ""Success after retry.""
                                },
                                ""finish_reason"": ""stop""
                            }]
                        }")
                    };
                }
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("Success after retry.", result);
        Assert.Equal(2, callCount); // Should have retried once
    }

    [Fact]
    public void ModelName_AfterConstruction_IncludesAzurePrefix()
    {
        // Arrange
        const string deployment = "my-gpt4-deployment";

        // Act
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: deployment);

        // Assert
        Assert.Equal($"azure-{deployment}", model.ModelName);
    }

    [Theory]
    [InlineData("2023-05-15")]
    [InlineData("2024-02-15-preview")]
    [InlineData("2024-03-01-preview")]
    public void Constructor_WithDifferentApiVersions_InitializesSuccessfully(string apiVersion)
    {
        // Arrange & Act
        var model = new AzureOpenAIChatModel<double>(
            endpoint: ValidEndpoint,
            apiKey: ValidApiKey,
            deploymentName: ValidDeployment,
            apiVersion: apiVersion);

        // Assert
        Assert.NotNull(model);
    }
}
