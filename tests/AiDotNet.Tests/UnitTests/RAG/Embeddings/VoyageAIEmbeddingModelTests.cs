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
    public class VoyageAIEmbeddingModelTests
    {
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

                return new HttpResponseMessage(_status) { Content = new StringContent(_responseBody ?? "") };
            }
        }

        private static VoyageAIEmbeddingModel<double> ModelWith(StubHandler handler, int dimension)
            => new("test-api-key", "voyage-3", "document", dimension, new HttpClient(handler));

        // ────────── Constructor validation ──────────

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 1024);
            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(16000, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentException()
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                new VoyageAIEmbeddingModel<double>(null, "voyage-3", "document", 1024));
            Assert.Contains("API key cannot be empty", ex.Message);
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentException()
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", null, "document", 1024));
            Assert.Contains("Model cannot be empty", ex.Message);
        }

        [Fact]
        public void Constructor_WithNullInputType_ThrowsArgumentException()
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", null, 1024));
            Assert.Contains("Input type cannot be empty", ex.Message);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 0));
            Assert.Contains("Dimension must be positive", ex.Message);
        }

        [Fact]
        public void MaxTokens_ReturnsCorrectValue()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 1024);
            Assert.Equal(16000, model.MaxTokens);
        }

        // ────────── Input validation (no network) ──────────

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 1024);
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 1024);
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 1024);
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 1024);
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(new List<string>()));
        }

        // ────────── Failure surfaces (no fake fallback) ──────────

        [Fact]
        public void Embed_WhenApiReturnsError_ThrowsInsteadOfFakeVector()
        {
            var handler = new StubHandler("{\"detail\":\"unauthorized\"}", HttpStatusCode.Unauthorized);
            var model = ModelWith(handler, 3);

            var ex = Assert.Throws<HttpRequestException>(() => model.Embed("hello"));
            Assert.Contains("Voyage AI API request failed", ex.Message);
        }

        [Fact]
        public void EmbedBatch_WhenApiReturnsError_ThrowsInsteadOfFakeVector()
        {
            var handler = new StubHandler("boom", HttpStatusCode.InternalServerError);
            var model = ModelWith(handler, 3);

            Assert.Throws<HttpRequestException>(() =>
                model.EmbedBatch(new List<string> { "a", "b" }));
        }

        // ────────── Real Voyage API endpoint/body/response ──────────

        [Fact]
        public void Embed_PostsToVoyageEndpointWithCorrectBodyAndParsesResponse()
        {
            var handler = new StubHandler(
                "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"embedding\":[0.1,0.2,0.3],\"index\":0}],\"model\":\"voyage-3\"}");
            var model = ModelWith(handler, 3);

            var embedding = model.Embed("hello world");

            // Parsed the real response
            Assert.Equal(3, embedding.Length);
            Assert.Equal(0.1, embedding[0], 6);
            Assert.Equal(0.2, embedding[1], 6);
            Assert.Equal(0.3, embedding[2], 6);

            // Hit the real Voyage endpoint
            Assert.Equal("https://api.voyageai.com/v1/embeddings", handler.LastRequestUri.ToString());

            // Correct request body: model, input[], input_type
            var body = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("voyage-3", (string)body["model"]);
            Assert.Equal("document", (string)body["input_type"]);
            Assert.Equal("hello world", (string)body["input"][0]);
        }

        [Fact]
        public async Task EmbedBatchAsync_PostsAllInputsAndReturnsMatrix()
        {
            var handler = new StubHandler(
                "{\"data\":[{\"embedding\":[0.1,0.2,0.3],\"index\":0},{\"embedding\":[0.4,0.5,0.6],\"index\":1}]}");
            var model = ModelWith(handler, 3);

            var matrix = await model.EmbedBatchAsync(new List<string> { "a", "b" });

            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(0.4, matrix[1, 0], 6);

            var body = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("a", (string)body["input"][0]);
            Assert.Equal("b", (string)body["input"][1]);
        }

        [Fact]
        public void Constructor_WithDifferentInputTypes_CreatesInstances()
        {
            var documentModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "document", 1024);
            var queryModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-3", "query", 1024);

            Assert.NotNull(documentModel);
            Assert.NotNull(queryModel);
        }
    }
}
