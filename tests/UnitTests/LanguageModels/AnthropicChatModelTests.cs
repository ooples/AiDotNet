using AiDotNet.LanguageModels;
using Xunit;
using System.Net;
using System.Net.Http;
using Moq;
using Moq.Protected;

namespace AiDotNetTests.UnitTests.LanguageModels;

/// <summary>
/// Unit tests for the AnthropicChatModel class.
/// </summary>
public class AnthropicChatModelTests
{
    [Fact]
    public void Constructor_WithValidApiKey_InitializesSuccessfully()
    {
        // Arrange & Act
        var model = new AnthropicChatModel<double>("test-api-key");

        // Assert
        Assert.NotNull(model);
        Assert.Equal("claude-3-sonnet-20240229", model.ModelName);
        Assert.Equal(200000, model.MaxContextTokens);
        Assert.Equal(4096, model.MaxGenerationTokens);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_WithInvalidApiKey_ThrowsArgumentException(string? apiKey)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>(apiKey!));
    }

    [Theory]
    [InlineData("claude-3-opus-20240229", 200000)]
    [InlineData("claude-3-haiku-20240307", 200000)]
    [InlineData("claude-2.1", 200000)]
    public void Constructor_WithDifferentModels_SetsCorrectContextWindow(string modelName, int expectedTokens)
    {
        // Arrange & Act
        var model = new AnthropicChatModel<double>("test-api-key", modelName: modelName);

        // Assert
        Assert.Equal(modelName, model.ModelName);
        Assert.Equal(expectedTokens, model.MaxContextTokens);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>("test-api-key", temperature: temperature));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>("test-api-key", topP: topP));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    public void Constructor_WithInvalidMaxTokens_ThrowsArgumentException(int maxTokens)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new AnthropicChatModel<double>("test-api-key", maxTokens: maxTokens));
    }

    [Fact]
    public async Task GenerateAsync_WithNullPrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new AnthropicChatModel<double>("test-api-key");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            model.GenerateAsync(null!));
    }

    [Fact]
    public async Task GenerateAsync_WithEmptyPrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new AnthropicChatModel<double>("test-api-key");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            model.GenerateAsync(""));
    }

    [Fact]
    public async Task GenerateAsync_WithWhitespacePrompt_ThrowsArgumentException()
    {
        // Arrange
        var model = new AnthropicChatModel<double>("test-api-key");

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
                    ""id"": ""msg_123"",
                    ""type"": ""message"",
                    ""role"": ""assistant"",
                    ""content"": [{
                        ""type"": ""text"",
                        ""text"": ""This is a test response from Claude.""
                    }],
                    ""model"": ""claude-3-sonnet-20240229"",
                    ""stop_reason"": ""end_turn"",
                    ""usage"": {
                        ""input_tokens"": 10,
                        ""output_tokens"": 20
                    }
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AnthropicChatModel<double>("test-api-key", httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("This is a test response from Claude.", result);
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
        var model = new AnthropicChatModel<double>("test-api-key", httpClient: httpClient);

        // Act & Assert
        await Assert.ThrowsAsync<HttpRequestException>(() =>
            model.GenerateAsync("Test prompt"));
    }

    [Fact]
    public async Task GenerateAsync_WithEmptyContent_ThrowsInvalidOperationException()
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
                    ""id"": ""msg_123"",
                    ""type"": ""message"",
                    ""role"": ""assistant"",
                    ""content"": [],
                    ""model"": ""claude-3-sonnet-20240229""
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AnthropicChatModel<double>("test-api-key", httpClient: httpClient);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            model.GenerateAsync("Test prompt"));
    }

    [Fact]
    public async Task GenerateAsync_WithNullContent_ThrowsInvalidOperationException()
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
                    ""id"": ""msg_123"",
                    ""type"": ""message"",
                    ""role"": ""assistant"",
                    ""model"": ""claude-3-sonnet-20240229""
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AnthropicChatModel<double>("test-api-key", httpClient: httpClient);

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
                    ""id"": ""msg_123"",
                    ""type"": ""message"",
                    ""role"": ""assistant"",
                    ""content"": [{
                        ""type"": ""text"",
                        ""text"": ""Synchronous response from Claude.""
                    }],
                    ""model"": ""claude-3-sonnet-20240229""
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AnthropicChatModel<double>("test-api-key", httpClient: httpClient);

        // Act
        var result = model.Generate("Test prompt");

        // Assert
        Assert.Equal("Synchronous response from Claude.", result);
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
                    ""id"": ""msg_123"",
                    ""type"": ""message"",
                    ""role"": ""assistant"",
                    ""content"": [{
                        ""type"": ""text"",
                        ""text"": ""Response via GenerateResponseAsync.""
                    }],
                    ""model"": ""claude-3-sonnet-20240229""
                }")
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AnthropicChatModel<double>("test-api-key", httpClient: httpClient);

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
                            ""id"": ""msg_123"",
                            ""type"": ""message"",
                            ""role"": ""assistant"",
                            ""content"": [{
                                ""type"": ""text"",
                                ""text"": ""Success after retry.""
                            }],
                            ""model"": ""claude-3-sonnet-20240229""
                        }")
                    };
                }
            });

        var httpClient = new HttpClient(mockHttpMessageHandler.Object);
        var model = new AnthropicChatModel<double>("test-api-key", httpClient: httpClient);

        // Act
        var result = await model.GenerateAsync("Test prompt");

        // Assert
        Assert.Equal("Success after retry.", result);
        Assert.Equal(2, callCount); // Should have retried once
    }

    [Fact]
    public void ModelName_AfterConstruction_MatchesProvidedName()
    {
        // Arrange
        const string expectedModel = "claude-3-opus-20240229";

        // Act
        var model = new AnthropicChatModel<double>("test-api-key", modelName: expectedModel);

        // Assert
        Assert.Equal(expectedModel, model.ModelName);
    }

    [Fact]
    public void MaxContextTokens_ForAllClaudeModels_Is200000()
    {
        // Arrange & Act
        var sonnet = new AnthropicChatModel<double>("test-api-key", modelName: "claude-3-sonnet-20240229");
        var opus = new AnthropicChatModel<double>("test-api-key", modelName: "claude-3-opus-20240229");
        var haiku = new AnthropicChatModel<double>("test-api-key", modelName: "claude-3-haiku-20240307");

        // Assert
        Assert.Equal(200000, sonnet.MaxContextTokens);
        Assert.Equal(200000, opus.MaxContextTokens);
        Assert.Equal(200000, haiku.MaxContextTokens);
    }
}
