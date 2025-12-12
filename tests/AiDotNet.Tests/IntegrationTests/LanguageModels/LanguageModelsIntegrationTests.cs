using AiDotNet.LanguageModels;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using Xunit;

namespace AiDotNetTests.IntegrationTests.LanguageModels
{
    /// <summary>
    /// Integration tests for language models with comprehensive coverage of
    /// API interactions, message formatting, error handling, and response parsing.
    /// Uses mocked HTTP responses to avoid real API calls.
    /// </summary>
    public class LanguageModelsIntegrationTests
    {
        #region Test Helpers

        /// <summary>
        /// Custom HttpMessageHandler for mocking HTTP responses in tests.
        /// </summary>
        private class MockHttpMessageHandler : HttpMessageHandler
        {
            private readonly Func<HttpRequestMessage, HttpResponseMessage> _responseFunc;

            public MockHttpMessageHandler(Func<HttpRequestMessage, HttpResponseMessage> responseFunc)
            {
                _responseFunc = responseFunc;
            }

            protected override Task<HttpResponseMessage> SendAsync(
                HttpRequestMessage request,
                CancellationToken cancellationToken)
            {
                return Task.FromResult(_responseFunc(request));
            }
        }

        private static HttpClient CreateMockHttpClient(HttpStatusCode statusCode, string responseContent)
        {
            var handler = new MockHttpMessageHandler(_ => new HttpResponseMessage
            {
                StatusCode = statusCode,
                Content = new StringContent(responseContent, Encoding.UTF8, "application/json")
            });
            return new HttpClient(handler);
        }

        private static HttpClient CreateMockHttpClientWithRequestValidator(
            HttpStatusCode statusCode,
            string responseContent,
            Action<HttpRequestMessage> requestValidator)
        {
            var handler = new MockHttpMessageHandler(request =>
            {
                requestValidator(request);
                return new HttpResponseMessage
                {
                    StatusCode = statusCode,
                    Content = new StringContent(responseContent, Encoding.UTF8, "application/json")
                };
            });
            return new HttpClient(handler);
        }

        #endregion

        #region OpenAIChatModel Integration Tests

        [Fact]
        public void OpenAIChatModel_Initialization_WithValidConfiguration_Succeeds()
        {
            // Arrange & Act
            var model = new OpenAIChatModel<double>(
                apiKey: "test-key-12345",
                modelName: "gpt-4",
                temperature: 0.8,
                maxTokens: 1024,
                topP: 0.9,
                frequencyPenalty: 0.5,
                presencePenalty: 0.3
            );

            // Assert
            Assert.NotNull(model);
            Assert.Equal("gpt-4", model.ModelName);
            Assert.Equal(8192, model.MaxContextTokens); // GPT-4 context window
            Assert.Equal(1024, model.MaxGenerationTokens);
        }

        [Fact]
        public void OpenAIChatModel_Initialization_WithDifferentModels_SetsCorrectContextLimits()
        {
            // Test different GPT models and their context windows
            var testCases = new[]
            {
                ("gpt-3.5-turbo", 4096),
                ("gpt-3.5-turbo-16k", 16384),
                ("gpt-4", 8192),
                ("gpt-4-32k", 32768),
                ("gpt-4-turbo", 128000),
                ("gpt-4o", 128000)
            };

            foreach (var (modelName, expectedContextTokens) in testCases)
            {
                // Arrange & Act
                var model = new OpenAIChatModel<double>("test-key", modelName: modelName);

                // Assert
                Assert.Equal(modelName, model.ModelName);
                Assert.Equal(expectedContextTokens, model.MaxContextTokens);
            }
        }

        [Fact]
        public async Task OpenAIChatModel_GenerateAsync_WithValidPrompt_FormatsRequestCorrectly()
        {
            // Arrange
            var expectedResponse = @"{
                ""id"": ""chatcmpl-123"",
                ""object"": ""chat.completion"",
                ""created"": 1677652288,
                ""model"": ""gpt-3.5-turbo"",
                ""choices"": [{
                    ""index"": 0,
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""The capital of France is Paris.""
                    },
                    ""finish_reason"": ""stop""
                }],
                ""usage"": {
                    ""prompt_tokens"": 10,
                    ""completion_tokens"": 20,
                    ""total_tokens"": 30
                }
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request =>
                {
                    capturedRequest = request;
                }
            );

            var model = new OpenAIChatModel<double>(
                apiKey: "test-key",
                modelName: "gpt-3.5-turbo",
                temperature: 0.7,
                maxTokens: 100,
                httpClient: httpClient
            );

            // Act
            var result = await model.GenerateAsync("What is the capital of France?");

            // Assert - Response parsing
            Assert.Equal("The capital of France is Paris.", result);

            // Assert - Request formatting
            Assert.NotNull(capturedRequest);
            Assert.Equal(HttpMethod.Post, capturedRequest.Method);
            Assert.Equal("https://api.openai.com/v1/chat/completions", capturedRequest.RequestUri?.ToString());

            // Verify Authorization header
            Assert.True(capturedRequest.Headers.Contains("Authorization"));
            var authHeader = capturedRequest.Headers.GetValues("Authorization").First();
            Assert.StartsWith("Bearer ", authHeader);
            Assert.Contains("test-key", authHeader);

            // Verify request body
            var requestBody = await capturedRequest.Content!.ReadAsStringAsync();
            Assert.Contains("\"model\":\"gpt-3.5-turbo\"", requestBody.Replace(" ", ""));
            Assert.Contains("\"temperature\":0.7", requestBody.Replace(" ", ""));
            Assert.Contains("\"max_tokens\":100", requestBody.Replace(" ", ""));
            Assert.Contains("What is the capital of France?", requestBody);
        }

        [Fact]
        public async Task OpenAIChatModel_GenerateAsync_WithCustomParameters_IncludesAllParametersInRequest()
        {
            // Arrange
            var expectedResponse = @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Test response""
                    }
                }]
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var model = new OpenAIChatModel<double>(
                apiKey: "test-key",
                temperature: 0.9,
                topP: 0.95,
                frequencyPenalty: 1.0,
                presencePenalty: 0.5,
                httpClient: httpClient
            );

            // Act
            await model.GenerateAsync("Test prompt");

            // Assert
            var requestBody = await capturedRequest!.Content!.ReadAsStringAsync();
            Assert.Contains("\"temperature\":0.9", requestBody.Replace(" ", ""));
            Assert.Contains("\"top_p\":0.95", requestBody.Replace(" ", ""));
            Assert.Contains("\"frequency_penalty\":1", requestBody.Replace(" ", ""));
            Assert.Contains("\"presence_penalty\":0.5", requestBody.Replace(" ", ""));
        }

        [Theory]
        [InlineData(HttpStatusCode.Unauthorized, "Invalid API key")]
        [InlineData(HttpStatusCode.BadRequest, "Invalid request format")]
        [InlineData(HttpStatusCode.TooManyRequests, "Rate limit exceeded")]
        [InlineData(HttpStatusCode.InternalServerError, "Internal server error")]
        public async Task OpenAIChatModel_GenerateAsync_WithErrorResponse_ThrowsHttpRequestException(
            HttpStatusCode statusCode, string errorMessage)
        {
            // Arrange
            var errorResponse = $@"{{
                ""error"": {{
                    ""message"": ""{errorMessage}"",
                    ""type"": ""invalid_request_error""
                }}
            }}";

            var httpClient = CreateMockHttpClient(statusCode, errorResponse);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            var exception = await Assert.ThrowsAsync<HttpRequestException>(
                () => model.GenerateAsync("Test prompt")
            );
            Assert.Contains(statusCode.ToString(), exception.Message);
        }

        [Fact]
        public async Task OpenAIChatModel_GenerateAsync_WithEmptyChoices_ThrowsInvalidOperationException()
        {
            // Arrange
            var emptyResponse = @"{
                ""id"": ""chatcmpl-123"",
                ""choices"": []
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, emptyResponse);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            var exception = await Assert.ThrowsAsync<InvalidOperationException>(
                () => model.GenerateAsync("Test prompt")
            );
            Assert.Contains("no choices", exception.Message);
        }

        [Fact]
        public async Task OpenAIChatModel_GenerateAsync_WithMissingContent_ThrowsInvalidOperationException()
        {
            // Arrange
            var invalidResponse = @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": """"
                    }
                }]
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, invalidResponse);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            var exception = await Assert.ThrowsAsync<InvalidOperationException>(
                () => model.GenerateAsync("Test prompt")
            );
            Assert.Contains("empty message content", exception.Message);
        }

        [Fact]
        public async Task OpenAIChatModel_GenerateAsync_WithLongPrompt_CalculatesTokenEstimateCorrectly()
        {
            // Arrange
            var longPrompt = new string('a', 20000); // ~5000 tokens estimated
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, "{}");
            var model = new OpenAIChatModel<double>("test-key", modelName: "gpt-3.5-turbo", httpClient: httpClient);

            // Act & Assert - Should throw because prompt exceeds 4096 token limit
            var exception = await Assert.ThrowsAsync<ArgumentException>(
                () => model.GenerateAsync(longPrompt)
            );
            Assert.Contains("too long", exception.Message);
            Assert.Contains("estimated tokens", exception.Message);
        }

        [Fact]
        public async Task OpenAIChatModel_GenerateResponseAsync_CallsGenerateAsync()
        {
            // Arrange
            var expectedResponse = @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Response via GenerateResponseAsync""
                    }
                }]
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, expectedResponse);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = await model.GenerateResponseAsync("Test prompt");

            // Assert
            Assert.Equal("Response via GenerateResponseAsync", result);
        }

        #endregion

        #region AnthropicChatModel Integration Tests

        [Fact]
        public void AnthropicChatModel_Initialization_WithValidConfiguration_Succeeds()
        {
            // Arrange & Act
            var model = new AnthropicChatModel<double>(
                apiKey: "test-anthropic-key",
                modelName: "claude-3-opus-20240229",
                temperature: 0.8,
                maxTokens: 2048,
                topP: 0.9,
                topK: 40
            );

            // Assert
            Assert.NotNull(model);
            Assert.Equal("claude-3-opus-20240229", model.ModelName);
            Assert.Equal(200000, model.MaxContextTokens); // Claude 3 context window
            Assert.Equal(2048, model.MaxGenerationTokens);
        }

        [Fact]
        public void AnthropicChatModel_Initialization_WithDifferentModels_SetsCorrectContextLimits()
        {
            // Test different Claude models and their context windows
            var testCases = new[]
            {
                ("claude-3-opus-20240229", 200000),
                ("claude-3-sonnet-20240229", 200000),
                ("claude-3-haiku-20240307", 200000),
                ("claude-2.1", 200000),
                ("claude-2.0", 100000)
            };

            foreach (var (modelName, expectedContextTokens) in testCases)
            {
                // Arrange & Act
                var model = new AnthropicChatModel<double>("test-key", modelName: modelName);

                // Assert
                Assert.Equal(modelName, model.ModelName);
                Assert.Equal(expectedContextTokens, model.MaxContextTokens);
            }
        }

        [Fact]
        public async Task AnthropicChatModel_GenerateAsync_WithValidPrompt_FormatsRequestCorrectly()
        {
            // Arrange
            var expectedResponse = @"{
                ""id"": ""msg_123"",
                ""type"": ""message"",
                ""role"": ""assistant"",
                ""content"": [{
                    ""type"": ""text"",
                    ""text"": ""Machine learning is a subset of artificial intelligence.""
                }],
                ""model"": ""claude-3-sonnet-20240229"",
                ""stop_reason"": ""end_turn"",
                ""usage"": {
                    ""input_tokens"": 15,
                    ""output_tokens"": 25
                }
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var model = new AnthropicChatModel<double>(
                apiKey: "test-anthropic-key",
                modelName: "claude-3-sonnet-20240229",
                temperature: 0.7,
                maxTokens: 1024,
                httpClient: httpClient
            );

            // Act
            var result = await model.GenerateAsync("What is machine learning?");

            // Assert - Response parsing
            Assert.Equal("Machine learning is a subset of artificial intelligence.", result);

            // Assert - Request formatting
            Assert.NotNull(capturedRequest);
            Assert.Equal(HttpMethod.Post, capturedRequest.Method);
            Assert.Equal("https://api.anthropic.com/v1/messages", capturedRequest.RequestUri?.ToString());

            // Verify headers
            Assert.True(capturedRequest.Headers.Contains("x-api-key"));
            var apiKeyHeader = capturedRequest.Headers.GetValues("x-api-key").First();
            Assert.Equal("test-anthropic-key", apiKeyHeader);

            Assert.True(capturedRequest.Headers.Contains("anthropic-version"));
            var versionHeader = capturedRequest.Headers.GetValues("anthropic-version").First();
            Assert.Equal("2023-06-01", versionHeader);

            // Verify request body
            var requestBody = await capturedRequest.Content!.ReadAsStringAsync();
            Assert.Contains("\"model\":\"claude-3-sonnet-20240229\"", requestBody.Replace(" ", ""));
            Assert.Contains("\"temperature\":0.7", requestBody.Replace(" ", ""));
            Assert.Contains("\"max_tokens\":1024", requestBody.Replace(" ", ""));
            Assert.Contains("What is machine learning?", requestBody);
        }

        [Fact]
        public async Task AnthropicChatModel_GenerateAsync_WithTopKParameter_IncludesInRequest()
        {
            // Arrange
            var expectedResponse = @"{
                ""content"": [{
                    ""type"": ""text"",
                    ""text"": ""Test response""
                }]
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var model = new AnthropicChatModel<double>(
                apiKey: "test-key",
                topK: 50,
                httpClient: httpClient
            );

            // Act
            await model.GenerateAsync("Test prompt");

            // Assert
            var requestBody = await capturedRequest!.Content!.ReadAsStringAsync();
            Assert.Contains("\"top_k\":50", requestBody.Replace(" ", ""));
        }

        [Fact]
        public async Task AnthropicChatModel_GenerateAsync_WithMultipleContentBlocks_CombinesTextCorrectly()
        {
            // Arrange
            var responseWithMultipleBlocks = @"{
                ""content"": [
                    {
                        ""type"": ""text"",
                        ""text"": ""First block of text.""
                    },
                    {
                        ""type"": ""text"",
                        ""text"": ""Second block of text.""
                    },
                    {
                        ""type"": ""text"",
                        ""text"": ""Third block of text.""
                    }
                ]
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, responseWithMultipleBlocks);
            var model = new AnthropicChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = await model.GenerateAsync("Test prompt");

            // Assert
            Assert.Contains("First block of text.", result);
            Assert.Contains("Second block of text.", result);
            Assert.Contains("Third block of text.", result);
            // Content blocks should be joined with newlines
            Assert.Contains("\n", result);
        }

        [Fact]
        public async Task AnthropicChatModel_GenerateAsync_WithEmptyContent_ThrowsInvalidOperationException()
        {
            // Arrange
            var emptyResponse = @"{
                ""id"": ""msg_123"",
                ""content"": []
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, emptyResponse);
            var model = new AnthropicChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            var exception = await Assert.ThrowsAsync<InvalidOperationException>(
                () => model.GenerateAsync("Test prompt")
            );
            Assert.Contains("no content", exception.Message);
        }

        [Fact]
        public async Task AnthropicChatModel_GenerateAsync_WithNonTextContent_ThrowsInvalidOperationException()
        {
            // Arrange
            var nonTextResponse = @"{
                ""content"": [{
                    ""type"": ""image"",
                    ""source"": ""base64data""
                }]
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, nonTextResponse);
            var model = new AnthropicChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            var exception = await Assert.ThrowsAsync<InvalidOperationException>(
                () => model.GenerateAsync("Test prompt")
            );
            Assert.Contains("no text content", exception.Message);
        }

        [Theory]
        [InlineData(-0.1)]
        [InlineData(1.1)]
        public void AnthropicChatModel_Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new AnthropicChatModel<double>("test-key", temperature: temperature)
            );
            Assert.Contains("Temperature must be between 0 and 1", exception.Message);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(4097)]
        public void AnthropicChatModel_Constructor_WithInvalidMaxTokens_ThrowsArgumentException(int maxTokens)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new AnthropicChatModel<double>("test-key", maxTokens: maxTokens)
            );
            Assert.Contains("Max tokens must be between 1 and 4096", exception.Message);
        }

        #endregion

        #region AzureOpenAIChatModel Integration Tests

        [Fact]
        public void AzureOpenAIChatModel_Initialization_WithValidConfiguration_Succeeds()
        {
            // Arrange & Act
            var model = new AzureOpenAIChatModel<double>(
                endpoint: "https://my-resource.openai.azure.com",
                apiKey: "test-azure-key",
                deploymentName: "gpt-4-deployment",
                apiVersion: "2024-02-15-preview",
                temperature: 0.7,
                maxTokens: 1024
            );

            // Assert
            Assert.NotNull(model);
            Assert.Equal("azure-gpt-4-deployment", model.ModelName);
            Assert.Equal(8192, model.MaxContextTokens);
            Assert.Equal(1024, model.MaxGenerationTokens);
        }

        [Fact]
        public async Task AzureOpenAIChatModel_GenerateAsync_WithValidPrompt_FormatsAzureEndpointCorrectly()
        {
            // Arrange
            var expectedResponse = @"{
                ""id"": ""chatcmpl-azure-123"",
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Azure OpenAI response""
                    },
                    ""finish_reason"": ""stop""
                }],
                ""usage"": {
                    ""prompt_tokens"": 10,
                    ""completion_tokens"": 5,
                    ""total_tokens"": 15
                }
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var model = new AzureOpenAIChatModel<double>(
                endpoint: "https://my-resource.openai.azure.com",
                apiKey: "test-azure-key",
                deploymentName: "gpt-35-turbo",
                apiVersion: "2024-02-15-preview",
                httpClient: httpClient
            );

            // Act
            var result = await model.GenerateAsync("Test Azure prompt");

            // Assert - Response
            Assert.Equal("Azure OpenAI response", result);

            // Assert - Request URL formatting
            Assert.NotNull(capturedRequest);
            var expectedUrl = "https://my-resource.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-15-preview";
            Assert.Equal(expectedUrl, capturedRequest.RequestUri?.ToString());

            // Verify api-key header (Azure uses different auth than OpenAI)
            Assert.True(capturedRequest.Headers.Contains("api-key"));
            var apiKeyHeader = capturedRequest.Headers.GetValues("api-key").First();
            Assert.Equal("test-azure-key", apiKeyHeader);

            // Should NOT have Authorization header (Azure uses api-key instead)
            Assert.False(capturedRequest.Headers.Contains("Authorization"));
        }

        [Fact]
        public async Task AzureOpenAIChatModel_GenerateAsync_WithTrailingSlashInEndpoint_HandlesCorrectly()
        {
            // Arrange
            var expectedResponse = @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Response""
                    }
                }]
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var model = new AzureOpenAIChatModel<double>(
                endpoint: "https://my-resource.openai.azure.com/",  // Trailing slash
                apiKey: "test-key",
                deploymentName: "gpt-4",
                httpClient: httpClient
            );

            // Act
            await model.GenerateAsync("Test");

            // Assert - Should handle trailing slash correctly (no double slashes)
            var url = capturedRequest!.RequestUri!.ToString();
            Assert.DoesNotContain("//openai", url);
            Assert.Contains("/openai/deployments/gpt-4/chat/completions", url);
        }

        [Fact]
        public void AzureOpenAIChatModel_Constructor_WithEmptyEndpoint_ThrowsArgumentException()
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new AzureOpenAIChatModel<double>(
                    endpoint: "",
                    apiKey: "test-key",
                    deploymentName: "gpt-4"
                )
            );
            Assert.Contains("Endpoint cannot be null or empty", exception.Message);
        }

        [Fact]
        public void AzureOpenAIChatModel_Constructor_WithEmptyDeploymentName_ThrowsArgumentException()
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new AzureOpenAIChatModel<double>(
                    endpoint: "https://test.openai.azure.com",
                    apiKey: "test-key",
                    deploymentName: ""
                )
            );
            Assert.Contains("Deployment name cannot be null or empty", exception.Message);
        }

        [Fact]
        public async Task AzureOpenAIChatModel_GenerateAsync_WithCustomApiVersion_IncludesInUrl()
        {
            // Arrange
            var expectedResponse = @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Response""
                    }
                }]
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var customApiVersion = "2023-12-01-preview";
            var model = new AzureOpenAIChatModel<double>(
                endpoint: "https://test.openai.azure.com",
                apiKey: "test-key",
                deploymentName: "test-deployment",
                apiVersion: customApiVersion,
                httpClient: httpClient
            );

            // Act
            await model.GenerateAsync("Test");

            // Assert
            var url = capturedRequest!.RequestUri!.ToString();
            Assert.Contains($"api-version={customApiVersion}", url);
        }

        #endregion

        #region ChatModelBase Integration Tests

        [Fact]
        public async Task ChatModelBase_GenerateAsync_WithNullPrompt_ThrowsArgumentException()
        {
            // Arrange
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, "{}");
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(
                () => model.GenerateAsync(null!)
            );
        }

        [Fact]
        public async Task ChatModelBase_GenerateAsync_WithEmptyPrompt_ThrowsArgumentException()
        {
            // Arrange
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, "{}");
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(
                () => model.GenerateAsync("")
            );
        }

        [Fact]
        public async Task ChatModelBase_GenerateAsync_WithWhitespacePrompt_ThrowsArgumentException()
        {
            // Arrange
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, "{}");
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(
                () => model.GenerateAsync("   \t\n")
            );
        }

        [Fact]
        public void ChatModelBase_Constructor_WithNegativeMaxContextTokens_ThrowsArgumentException()
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new TestChatModel(maxContextTokens: -1, maxGenerationTokens: 100)
            );
            Assert.Contains("Maximum context tokens must be positive", exception.Message);
        }

        [Fact]
        public void ChatModelBase_Constructor_WithNegativeMaxGenerationTokens_ThrowsArgumentException()
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new TestChatModel(maxContextTokens: 1000, maxGenerationTokens: -1)
            );
            Assert.Contains("Maximum generation tokens must be positive", exception.Message);
        }

        [Fact]
        public void ChatModelBase_ValidateApiKey_WithNullKey_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(
                () => new OpenAIChatModel<double>(null!)
            );
        }

        [Fact]
        public void ChatModelBase_ValidateApiKey_WithEmptyKey_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => new OpenAIChatModel<double>("")
            );
        }

        [Fact]
        public void ChatModelBase_ValidateApiKey_WithWhitespaceKey_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => new OpenAIChatModel<double>("   ")
            );
        }

        /// <summary>
        /// Test implementation of ChatModelBase for testing base functionality.
        /// </summary>
        private class TestChatModel : ChatModelBase<double>
        {
            public TestChatModel(int maxContextTokens, int maxGenerationTokens)
                : base(null, maxContextTokens, maxGenerationTokens)
            {
            }

            protected override Task<string> GenerateAsyncCore(string prompt)
            {
                return Task.FromResult("Test response");
            }
        }

        #endregion

        #region Parameter Validation Integration Tests

        [Theory]
        [InlineData(-0.1)]
        [InlineData(2.1)]
        public void OpenAIChatModel_Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new OpenAIChatModel<double>("test-key", temperature: temperature)
            );
            Assert.Contains("Temperature must be between 0 and 2", exception.Message);
        }

        [Theory]
        [InlineData(-0.1)]
        [InlineData(1.1)]
        public void OpenAIChatModel_Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new OpenAIChatModel<double>("test-key", topP: topP)
            );
            Assert.Contains("TopP must be between 0 and 1", exception.Message);
        }

        [Theory]
        [InlineData(-2.1)]
        [InlineData(2.1)]
        public void OpenAIChatModel_Constructor_WithInvalidFrequencyPenalty_ThrowsArgumentException(double penalty)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new OpenAIChatModel<double>("test-key", frequencyPenalty: penalty)
            );
            Assert.Contains("Frequency penalty must be between -2 and 2", exception.Message);
        }

        [Theory]
        [InlineData(-2.1)]
        [InlineData(2.1)]
        public void OpenAIChatModel_Constructor_WithInvalidPresencePenalty_ThrowsArgumentException(double penalty)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new OpenAIChatModel<double>("test-key", presencePenalty: penalty)
            );
            Assert.Contains("Presence penalty must be between -2 and 2", exception.Message);
        }

        [Theory]
        [InlineData(-0.1)]
        [InlineData(1.1)]
        public void AnthropicChatModel_Constructor_WithInvalidTopP_ThrowsArgumentException(double topP)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new AnthropicChatModel<double>("test-key", topP: topP)
            );
            Assert.Contains("TopP must be between 0 and 1", exception.Message);
        }

        [Fact]
        public void AnthropicChatModel_Constructor_WithNegativeTopK_ThrowsArgumentException()
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new AnthropicChatModel<double>("test-key", topK: -1)
            );
            Assert.Contains("TopK must be non-negative", exception.Message);
        }

        [Theory]
        [InlineData(-0.1)]
        [InlineData(2.1)]
        public void AzureOpenAIChatModel_Constructor_WithInvalidTemperature_ThrowsArgumentException(double temperature)
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(
                () => new AzureOpenAIChatModel<double>(
                    "https://test.openai.azure.com",
                    "test-key",
                    "deployment",
                    temperature: temperature
                )
            );
            Assert.Contains("Temperature must be between 0 and 2", exception.Message);
        }

        #endregion

        #region Message Formatting and Response Parsing Tests

        [Fact]
        public async Task OpenAIChatModel_MessageFormatting_UsesUserRole()
        {
            // Arrange
            var expectedResponse = @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Response""
                    }
                }]
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act
            await model.GenerateAsync("User prompt");

            // Assert
            var requestBody = await capturedRequest!.Content!.ReadAsStringAsync();
            Assert.Contains("\"role\":\"user\"", requestBody.Replace(" ", ""));
            Assert.Contains("\"content\":\"User prompt\"", requestBody.Replace(" ", ""));
        }

        [Fact]
        public async Task AnthropicChatModel_MessageFormatting_UsesUserRole()
        {
            // Arrange
            var expectedResponse = @"{
                ""content"": [{
                    ""type"": ""text"",
                    ""text"": ""Response""
                }]
            }";

            HttpRequestMessage? capturedRequest = null;
            var httpClient = CreateMockHttpClientWithRequestValidator(
                HttpStatusCode.OK,
                expectedResponse,
                request => { capturedRequest = request; }
            );

            var model = new AnthropicChatModel<double>("test-key", httpClient: httpClient);

            // Act
            await model.GenerateAsync("User prompt");

            // Assert
            var requestBody = await capturedRequest!.Content!.ReadAsStringAsync();
            Assert.Contains("\"role\":\"user\"", requestBody.Replace(" ", ""));
            Assert.Contains("\"content\":\"User prompt\"", requestBody.Replace(" ", ""));
        }

        [Fact]
        public async Task OpenAIChatModel_ResponseParsing_ExtractsAssistantMessage()
        {
            // Arrange
            var complexResponse = @"{
                ""id"": ""chatcmpl-123"",
                ""object"": ""chat.completion"",
                ""created"": 1677652288,
                ""model"": ""gpt-3.5-turbo"",
                ""choices"": [
                    {
                        ""index"": 0,
                        ""message"": {
                            ""role"": ""assistant"",
                            ""content"": ""Here is a detailed explanation of machine learning algorithms.""
                        },
                        ""finish_reason"": ""stop""
                    }
                ],
                ""usage"": {
                    ""prompt_tokens"": 20,
                    ""completion_tokens"": 50,
                    ""total_tokens"": 70
                }
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, complexResponse);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = await model.GenerateAsync("Explain ML");

            // Assert
            Assert.Equal("Here is a detailed explanation of machine learning algorithms.", result);
        }

        [Fact]
        public async Task AnthropicChatModel_ResponseParsing_ExtractsTextFromContentBlocks()
        {
            // Arrange
            var complexResponse = @"{
                ""id"": ""msg_456"",
                ""type"": ""message"",
                ""role"": ""assistant"",
                ""content"": [
                    {
                        ""type"": ""text"",
                        ""text"": ""Neural networks are inspired by biological neurons.""
                    }
                ],
                ""model"": ""claude-3-sonnet-20240229"",
                ""stop_reason"": ""end_turn"",
                ""usage"": {
                    ""input_tokens"": 15,
                    ""output_tokens"": 30
                }
            }";

            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, complexResponse);
            var model = new AnthropicChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = await model.GenerateAsync("Explain neural networks");

            // Assert
            Assert.Equal("Neural networks are inspired by biological neurons.", result);
        }

        #endregion

        #region Token Counting and Limits Tests

        [Fact]
        public async Task ChatModelBase_EstimateTokenCount_ApproximatelyOneTokenPerFourChars()
        {
            // Arrange
            var prompt = new string('x', 400); // ~100 tokens
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, @"{
                ""choices"": [{""message"": {""content"": ""Response""}}]
            }");
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act - Should not throw (well within 4096 limit)
            await model.GenerateAsync(prompt);

            // Assert - Test passes if no exception
            Assert.True(true);
        }

        [Fact]
        public async Task OpenAIChatModel_GenerateAsync_ExceedingContextWindow_ThrowsArgumentException()
        {
            // Arrange
            var tooLongPrompt = new string('x', 50000); // Way over 4096 tokens for gpt-3.5-turbo
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, "{}");
            var model = new OpenAIChatModel<double>("test-key", modelName: "gpt-3.5-turbo", httpClient: httpClient);

            // Act & Assert
            var exception = await Assert.ThrowsAsync<ArgumentException>(
                () => model.GenerateAsync(tooLongPrompt)
            );
            Assert.Contains("too long", exception.Message);
            Assert.Contains("4096", exception.Message);
        }

        [Fact]
        public async Task OpenAIChatModel_LargeContextModel_AllowsLongerPrompts()
        {
            // Arrange
            var longPrompt = new string('x', 50000); // ~12,500 tokens - OK for gpt-4-turbo (128k)
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, @"{
                ""choices"": [{""message"": {""content"": ""Response""}}]
            }");
            var model = new OpenAIChatModel<double>("test-key", modelName: "gpt-4-turbo", httpClient: httpClient);

            // Act - Should not throw (within 128k limit)
            var result = await model.GenerateAsync(longPrompt);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Error Handling and Retry Logic Tests

        [Fact]
        public async Task ChatModelBase_RetryLogic_WithTransientError_RetriesRequest()
        {
            // Arrange
            int attemptCount = 0;
            var handler = new MockHttpMessageHandler(request =>
            {
                attemptCount++;
                if (attemptCount < 2) // Fail first time
                {
                    return new HttpResponseMessage
                    {
                        StatusCode = HttpStatusCode.ServiceUnavailable,
                        Content = new StringContent("{\"error\": \"Service temporarily unavailable\"}")
                    };
                }
                // Succeed on retry
                return new HttpResponseMessage
                {
                    StatusCode = HttpStatusCode.OK,
                    Content = new StringContent(@"{
                        ""choices"": [{
                            ""message"": {
                                ""role"": ""assistant"",
                                ""content"": ""Success after retry""
                            }
                        }]
                    }")
                };
            });

            var httpClient = new HttpClient(handler);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = await model.GenerateAsync("Test prompt");

            // Assert
            Assert.Equal("Success after retry", result);
            Assert.Equal(2, attemptCount); // Should have retried once
        }

        [Fact]
        public async Task ChatModelBase_RetryLogic_WithRateLimitError_RetriesRequest()
        {
            // Arrange
            int attemptCount = 0;
            var handler = new MockHttpMessageHandler(request =>
            {
                attemptCount++;
                if (attemptCount < 2)
                {
                    return new HttpResponseMessage
                    {
                        StatusCode = HttpStatusCode.TooManyRequests,
                        Content = new StringContent("{\"error\": \"Rate limit exceeded\"}")
                    };
                }
                return new HttpResponseMessage
                {
                    StatusCode = HttpStatusCode.OK,
                    Content = new StringContent(@"{
                        ""choices"": [{""message"": {""content"": ""Success""}}]
                    }")
                };
            });

            var httpClient = new HttpClient(handler);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = await model.GenerateAsync("Test prompt");

            // Assert
            Assert.Equal("Success", result);
            Assert.True(attemptCount >= 2);
        }

        [Fact]
        public async Task ChatModelBase_RetryLogic_WithPermanentError_DoesNotRetry()
        {
            // Arrange
            int attemptCount = 0;
            var handler = new MockHttpMessageHandler(request =>
            {
                attemptCount++;
                return new HttpResponseMessage
                {
                    StatusCode = HttpStatusCode.BadRequest, // Non-retryable
                    Content = new StringContent("{\"error\": \"Invalid request\"}")
                };
            });

            var httpClient = new HttpClient(handler);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            await Assert.ThrowsAsync<HttpRequestException>(
                () => model.GenerateAsync("Test prompt")
            );

            // Should only attempt once (no retries for 400)
            Assert.Equal(1, attemptCount);
        }

        [Fact]
        public async Task ChatModelBase_RetryLogic_WithInvalidJson_ThrowsWithoutRetry()
        {
            // Arrange
            int attemptCount = 0;
            var handler = new MockHttpMessageHandler(request =>
            {
                attemptCount++;
                return new HttpResponseMessage
                {
                    StatusCode = HttpStatusCode.OK,
                    Content = new StringContent("This is not valid JSON")
                };
            });

            var httpClient = new HttpClient(handler);
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act & Assert
            await Assert.ThrowsAsync<InvalidOperationException>(
                () => model.GenerateAsync("Test prompt")
            );

            // Should not retry JSON errors
            Assert.Equal(1, attemptCount);
        }

        #endregion

        #region Synchronous Method Tests

        [Fact]
        public void OpenAIChatModel_Generate_SynchronousMethod_Works()
        {
            // Arrange
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Synchronous response""
                    }
                }]
            }");
            var model = new OpenAIChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = model.Generate("Test prompt");

            // Assert
            Assert.Equal("Synchronous response", result);
        }

        [Fact]
        public void AnthropicChatModel_Generate_SynchronousMethod_Works()
        {
            // Arrange
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, @"{
                ""content"": [{
                    ""type"": ""text"",
                    ""text"": ""Synchronous response""
                }]
            }");
            var model = new AnthropicChatModel<double>("test-key", httpClient: httpClient);

            // Act
            var result = model.Generate("Test prompt");

            // Assert
            Assert.Equal("Synchronous response", result);
        }

        [Fact]
        public void AzureOpenAIChatModel_Generate_SynchronousMethod_Works()
        {
            // Arrange
            var httpClient = CreateMockHttpClient(HttpStatusCode.OK, @"{
                ""choices"": [{
                    ""message"": {
                        ""role"": ""assistant"",
                        ""content"": ""Azure synchronous response""
                    }
                }]
            }");
            var model = new AzureOpenAIChatModel<double>(
                "https://test.openai.azure.com",
                "test-key",
                "deployment",
                httpClient: httpClient
            );

            // Act
            var result = model.Generate("Test prompt");

            // Assert
            Assert.Equal("Azure synchronous response", result);
        }

        #endregion
    }
}
