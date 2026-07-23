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
    public class HuggingFaceEmbeddingModelTests
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

        private static HuggingFaceEmbeddingModel<double> ModelWith(StubHandler handler, int dimension)
            => new("sentence-transformers/all-MiniLM-L6-v2", "test-api-key", dimension, 512, new HttpClient(handler));

        // ────────── Constructor validation ──────────

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new HuggingFaceEmbeddingModel<double>("sentence-transformers/all-MiniLM-L6-v2", "test-api-key", 768, 512);
            Assert.NotNull(model);
            Assert.Equal(768, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithEmptyApiKey_CreatesInstance()
        {
            var model = new HuggingFaceEmbeddingModel<double>("sentence-transformers/all-MiniLM-L6-v2", "");
            Assert.NotNull(model);
        }

        [Fact]
        public void Constructor_WithNullModelName_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new HuggingFaceEmbeddingModel<double>(null, "test-api-key"));
            Assert.Contains("Model name cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new HuggingFaceEmbeddingModel<double>("sentence-transformers/all-MiniLM-L6-v2", "test-api-key", 0, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeMaxTokens_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new HuggingFaceEmbeddingModel<double>("sentence-transformers/all-MiniLM-L6-v2", "test-api-key", 768, -1));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        // ────────── Input validation (no network) ──────────

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new HuggingFaceEmbeddingModel<double>("sentence-transformers/all-MiniLM-L6-v2");
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new HuggingFaceEmbeddingModel<double>("sentence-transformers/all-MiniLM-L6-v2");
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new HuggingFaceEmbeddingModel<double>("sentence-transformers/all-MiniLM-L6-v2");
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(new List<string>()));
        }

        // ────────── Failure surfaces (no fake fallback) ──────────

        [Fact]
        public void Embed_WhenApiReturnsError_ThrowsInsteadOfFakeVector()
        {
            var handler = new StubHandler("{\"error\":\"model loading\"}", HttpStatusCode.ServiceUnavailable);
            var model = ModelWith(handler, 3);

            var ex = Assert.Throws<HttpRequestException>(() => model.Embed("hello"));
            Assert.Contains("HuggingFace API request failed", ex.Message);
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
        public void Embed_WithMockedSuccess_ParsesVectorAndPostsToInferenceEndpoint()
        {
            var handler = new StubHandler("[[0.1,0.2,0.3]]");
            var model = ModelWith(handler, 3);

            var embedding = model.Embed("hello world");

            Assert.Equal(3, embedding.Length);
            Assert.Equal(0.3, embedding[2], 6);
            Assert.Contains("api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                handler.LastRequestUri.ToString());
        }

        [Fact]
        public async Task EmbedBatchAsync_WithMockedSuccess_ReturnsMatrix()
        {
            var handler = new StubHandler("[[0.1,0.2,0.3],[0.4,0.5,0.6]]");
            var model = ModelWith(handler, 3);

            var matrix = await model.EmbedBatchAsync(new List<string> { "a", "b" });

            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(0.5, matrix[1, 1], 6);
        }
    }
}
