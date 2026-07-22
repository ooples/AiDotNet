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
    public class GooglePalmEmbeddingModelTests
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

        private static GooglePalmEmbeddingModel<double> ModelWith(StubHandler handler, int dimension)
            => new("test-project-id", "us-central1", "text-embedding-004", "test-api-key", dimension, new HttpClient(handler));

        // ────────── Constructor validation ──────────

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
            Assert.NotNull(model);
            Assert.Equal(768, model.EmbeddingDimension);
            Assert.Equal(2048, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullProjectId_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>(null, "us-central1", "textembedding-gecko@001", "test-api-key"));
        }

        [Fact]
        public void Constructor_WithNullLocation_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>("test-project-id", null, "textembedding-gecko@001", "test-api-key"));
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", null, "test-api-key"));
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", null));
        }

        // ────────── Input validation (no network) ──────────

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(new List<string>()));
        }

        // ────────── Failure surfaces (no fake fallback) ──────────

        [Fact]
        public void Embed_WhenApiReturnsError_ThrowsInsteadOfFakeVector()
        {
            var handler = new StubHandler("{\"error\":\"denied\"}", HttpStatusCode.Forbidden);
            var model = ModelWith(handler, 3);

            var ex = Assert.Throws<HttpRequestException>(() => model.Embed("hello"));
            Assert.Contains("Vertex AI API request failed", ex.Message);
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
        public void Embed_WithMockedSuccess_ParsesVectorAndPostsToVertexEndpoint()
        {
            var handler = new StubHandler("{\"predictions\":[{\"embeddings\":{\"values\":[0.1,0.2,0.3]}}]}");
            var model = ModelWith(handler, 3);

            var embedding = model.Embed("hello world");

            Assert.Equal(3, embedding.Length);
            Assert.Equal(0.2, embedding[1], 6);
            Assert.Contains("aiplatform.googleapis.com", handler.LastRequestUri.ToString());
            Assert.Contains("text-embedding-004:predict", handler.LastRequestUri.ToString());

            var body = JObject.Parse(handler.LastRequestBody);
            Assert.Equal("hello world", (string)body["instances"][0]["content"]);
        }

        [Fact]
        public async Task EmbedBatchAsync_WithMockedSuccess_ReturnsMatrix()
        {
            var handler = new StubHandler("{\"predictions\":[{\"embeddings\":{\"values\":[0.1,0.2,0.3]}},{\"embeddings\":{\"values\":[0.4,0.5,0.6]}}]}");
            var model = ModelWith(handler, 3);

            var matrix = await model.EmbedBatchAsync(new List<string> { "a", "b" });

            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(0.6, matrix[1, 2], 6);
        }
    }
}
