using AiDotNet.LanguageModels;
using Xunit;
using System.Net;
using System.Net.Http;
using Moq;
using Moq.Protected;

namespace AiDotNetTests.UnitTests.LanguageModels;

/// <summary>
/// Unit tests for the OpenAIChatModel class.
/// </summary>
public class OpenAIChatModelTests
{
    [Fact]
    public void Constructor_WithValidApiKey_InitializesSuccessfully()
    {
        // Arrange & Act
        var model = new OpenAIChatModel<double>("test-api-key");

        // Assert
        Assert.NotNull(model);
        Assert.Equal("gpt-3.5-turbo", model.ModelName);
        Assert.Equal(4096, model.MaxContextTokens);
        Assert.Equal(2048, model.MaxGenerationTokens);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_WithInvalidApiKey_ThrowsArgumentException(string? apiKey)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>(apiKey!));
    }

    [Theory]
    [InlineData("gpt-4", 8192)]
    [InlineData("gpt-4-turbo", 128000)]
    [InlineData("gpt-3.5-turbo-16k", 16384)]
    public void Constructor_WithDifferentModels_SetsCorrectContextWindow(string modelName, int expectedTokens)
    {
        // Arrange & Act
        var model = new OpenAIChatModel<double>("test-api-key", modelName: modelName);

        // Assert
        Assert.Equal(modelName, model.ModelName);
        Assert.Equal(expectedTokens, model.MaxContextTokens);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(2.1)]
    public void Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>("test-api-key", temperature: temperature));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>("test-api-key", topP: topP));
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    public void Constructor_WithInvalidFrequencyPenalty_ThrowsArgumentException(double penalty)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>("test-api-key", frequencyPenalty: penalty));
    }

    [Theory]
    [InlineData(-2.1)]
    [InlineData(2.1)]
    public void Constructor_WithInvalidPresencePenalty_ThrowsArgumentException(double penalty)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new OpenAIChatModel<double>("test-api-key", presencePenalty: penalty));
    }

    [Fact]
    public async Task GenerateAsync_WithNullPrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new OpenAIChatModel<double>("test-api-key");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            model.GenerateAsync(null!));
    }

    [Fact]
    public async Task GenerateAsync_WithEmptyPrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new OpenAIChatModel<double>("test-api-key");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            model.GenerateAsync(""));
    }

    [Fact]
    public async Task GenerateAsync_WithWhitespacePrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new OpenAIChatModel<double>("test-api-key");

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
                            ""content"": ""This is a test response.""
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
        var model = new OpenAIChatModel<double>("test-api-key", httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("This is a test response.", result);
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
        var model = new OpenAIChatModel<double>("test-api-key", httpClient: httpClient);

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
        var model = new OpenAIChatModel<double>("test-api-key", httpClient: httpClient);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            model.GenerateAsync("Test prompt"));
    }

    [Fact]
    public void Generate_SyncMethod_CallsAsyncAndReturnsResult()
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
                            ""content"": ""Sync response""
                        },
                        ""finish_reason"": ""stop""
                    }]
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new OpenAIChatModel<double>("test-api-key", httpClient: httpClient);

        // Act
        var result = model.Generate("Test prompt");

        // Assert
        Assert.Equal("Sync response", result);
    }

    [Fact]
    public async Task GenerateResponseAsync_AliasMethod_CallsGenerateAsync()
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
                        ""message"": {
                            ""content"": ""Alias response""
                        }
                    }]
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new OpenAIChatModel<double>("test-api-key", httpClient: httpClient);

        // Act
        var result = await model.GenerateResponseAsync("Test prompt");

        // Assert
        Assert.Equal("Alias response", result);
    }

    [Fact]
    public void Constructor_WithCustomParameters_SetsAllParameters()
    {
        // Arrange & Act
        var model = new OpenAIChatModel<double>(
            apiKey: "test-key",
            modelName: "gpt-4",
            temperature: 0.5,
            maxTokens: 1000,
            topP: 0.9,
            frequencyPenalty: 0.5,
            presencePenalty: 0.3);

        // Assert
        Assert.Equal("gpt-4", model.ModelName);
        Assert.Equal(1000, model.MaxGenerationTokens);
    }

    [Fact]
    public void Constructor_WithCustomEndpoint_UsesCustomEndpoint()
    {
        // Arrange & Act
        var model = new OpenAIChatModel<double>(
            apiKey: "test-key",
            endpoint: "https://custom-endpoint.com/v1/chat/completions");

        // Assert
        Assert.NotNull(model);
        // The endpoint is internal, but we verify initialization doesn't fail
    }

    [Fact]
    public async Task GenerateAsync_SetsAuthorizationHeader()
    {
        // Arrange
        HttpRequestMessage? capturedRequest = null;

        var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
        mockHttpMessageHandler
            .Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .Callback<HttpRequestMessage, CancellationToken>((req, _) => capturedRequest = req)
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(@"{
                    ""choices"": [{
                        ""message"": {
                            ""content"": ""test""
                        }
                    }]
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new OpenAIChatModel<double>("test-api-key-123", httpClient: httpClient);

        // Act
        await model.GenerateAsync("Test");

        // Assert
        Assert.NotNull(capturedRequest);
        Assert.NotNull(capturedRequest.Headers.Authorization);
        Assert.Equal("Bearer", capturedRequest.Headers.Authorization.Scheme);
        Assert.Equal("test-api-key-123", capturedRequest.Headers.Authorization.Parameter);
    }
}
