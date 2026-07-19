#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class OpenAIEmbeddingModelTests
    {
        /// <summary>Captures the outbound request and returns a canned response / status.</summary>
        private sealed class StubHandler : HttpMessageHandler
        {
            private readonly HttpStatusCode _status;
            private readonly string _responseBody;
            public string LastRequestBody { get; private set; } = "";
            public Uri LastRequestUri { get; private set; }

            public StubHandler(string responseBody, HttpStatusCode status = HttpStatusCode.OK)
            {
                _responseBody = responseBody;
                _status = status;
            }

            protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                LastRequestUri = request.RequestUri;
                if (request.Content is not null)
                    LastRequestBody = await request.Content.ReadAsStringAsync().ConfigureAwait(false);

                return new HttpResponseMessage(_status)
                {
                    Content = new StringContent(_responseBody ?? "")
                };
            }
        }

        private static OpenAIEmbeddingModel<double> ModelWith(StubHandler handler, int dimension)
            => new("test-api-key", "text-embedding-ada-002", dimension, 8191, new HttpClient(handler));

        // ────────── Constructor validation ──────────

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
            Assert.NotNull(model);
            Assert.Equal(1536, model.EmbeddingDimension);
            Assert.Equal(8191, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>(null, "text-embedding-ada-002"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithEmptyApiKey_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("", "text-embedding-ada-002"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithNullModelName_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", null));
            Assert.Contains("Model name cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 0, 8191));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeMaxTokens_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, -1));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        // ────────── Input validation (no network) ──────────

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new OpenAIEmbeddingModel<double>("test-api-key");
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            var model = new OpenAIEmbeddingModel<double>("test-api-key");
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new OpenAIEmbeddingModel<double>("test-api-key");
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new OpenAIEmbeddingModel<double>("test-api-key");
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(new List<string>()));
        }

        // ────────── Failure surfaces (no fake fallback) ──────────

        [Fact]
        public void Embed_WhenApiReturnsError_ThrowsInsteadOfFakeVector()
        {
            var handler = new StubHandler("{\"error\":\"unauthorized\"}", HttpStatusCode.Unauthorized);
            var model = ModelWith(handler, 3);

            var ex = Assert.Throws<HttpRequestException>(() => model.Embed("hello"));
            Assert.Contains("OpenAI API request failed", ex.Message);
        }

        [Fact]
        public void EmbedBatch_WhenApiReturnsError_ThrowsInsteadOfFakeVector()
        {
            var handler = new StubHandler("boom", HttpStatusCode.InternalServerError);
            var model = ModelWith(handler, 3);

            Assert.Throws<HttpRequestException>(() =>
                model.EmbedBatch(new List<string> { "a", "b" }));
        }

        // ────────── Success path via mocked handler ──────────

        [Fact]
        public void Embed_WithMockedSuccess_ParsesVectorAndPostsToOpenAIEndpoint()
        {
            var handler = new StubHandler("{\"data\":[{\"embedding\":[0.1,0.2,0.3]}]}");
            var model = ModelWith(handler, 3);

            var embedding = model.Embed("hello world");

            Assert.Equal(3, embedding.Length);
            Assert.Equal(0.1, embedding[0], 6);
            Assert.Equal(0.2, embedding[1], 6);
            Assert.Equal(0.3, embedding[2], 6);
            Assert.Equal("https://api.openai.com/v1/embeddings", handler.LastRequestUri.ToString());

            var body = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("text-embedding-ada-002", (string)body["model"]);
            Assert.Equal("hello world", (string)body["input"][0]);
        }

        [Fact]
        public async Task EmbedBatchAsync_WithMockedSuccess_ReturnsMatrix()
        {
            var handler = new StubHandler("{\"data\":[{\"embedding\":[0.1,0.2,0.3]},{\"embedding\":[0.4,0.5,0.6]}]}");
            var model = ModelWith(handler, 3);

            var matrix = await model.EmbedBatchAsync(new List<string> { "a", "b" });

            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(0.4, matrix[1, 0], 6);
        }
    }
}
