#nullable disable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class VoyageAIEmbeddingModelTests
    {
        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(16000, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>(null, "voyage-model-path.onnx", "document", 1024));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", null, "document", 1024));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullInputType_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", null, 1024));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithValidText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("This is a test sentence."));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSameTextTwice_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSingleText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_ReturnsNormalizedVector_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test normalization"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithEmptyText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithValidTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string>();

            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithFloatType_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<float>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test with float type"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithCustomDimension_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Testing custom dimension"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_Deterministic_MultipleInstances_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Determinism test"));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_AllRowsAreNormalized_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithLongText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var longText = string.Join(" ", Enumerable.Repeat("word", 2000));

            Assert.Throws<FileNotFoundException>(() => model.Embed(longText));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDifferentInputTypes_CreatesInstances()
        {
            var documentModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var queryModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "query", 1024);

            Assert.NotNull(documentModel);
            Assert.NotNull(queryModel);
        }

        [Fact(Timeout = 60000)]
        public async Task MaxTokens_ReturnsCorrectValue()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Equal(16000, model.MaxTokens);
        }

        // ────────── Success-path tests via mock embedding model ──────────
        // These test the EmbeddingModelBase behavior (dimension, normalization, batch consistency)
        // without requiring an actual ONNX model file.

        [Fact(Timeout = 60000)]
        public async Task MockEmbed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            using var model = new MockEmbeddingModel(1024);

            var embedding = model.Embed("Test sentence");

            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task MockEmbed_ReturnsNormalizedVector()
        {
            using var model = new MockEmbeddingModel(512);

            var embedding = model.Embed("Test normalization");

            var magnitude = 0.0;
            for (int i = 0; i < embedding.Length; i++)
                magnitude += embedding[i] * embedding[i];
            magnitude = Math.Sqrt(magnitude);
            Assert.Equal(1.0, magnitude, 5);
        }

        [Fact(Timeout = 60000)]
        public async Task MockEmbed_WithSameText_ReturnsSameEmbedding()
        {
            using var model = new MockEmbeddingModel(384);

            var embedding1 = model.Embed("Hello world");
            var embedding2 = model.Embed("Hello world");

            for (int i = 0; i < embedding1.Length; i++)
                Assert.Equal(embedding1[i], embedding2[i], 10);
        }

        [Fact(Timeout = 60000)]
        public async Task MockEmbed_WithDifferentTexts_ReturnsDifferentEmbeddings()
        {
            using var model = new MockEmbeddingModel(384);

            var embedding1 = model.Embed("Hello world");
            var embedding2 = model.Embed("Goodbye world");

            var hasDifference = false;
            for (int i = 0; i < embedding1.Length; i++)
            {
                if (Math.Abs(embedding1[i] - embedding2[i]) > 1e-10)
                {
                    hasDifference = true;
                    break;
                }
            }
            Assert.True(hasDifference, "Embeddings for different texts should be different");
        }

        [Fact(Timeout = 60000)]
        public async Task MockEmbedBatch_ReturnsCorrectDimensions()
        {
            using var model = new MockEmbeddingModel(256);
            var texts = new List<string> { "First", "Second", "Third" };

            var embeddings = model.EmbedBatch(texts);

            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(256, embeddings.Columns);
        }

        [Fact(Timeout = 60000)]
        public async Task MockEmbedBatch_ProducesSameAsIndividualCalls()
        {
            using var model = new MockEmbeddingModel(384);
            var texts = new List<string> { "Alpha", "Beta", "Gamma" };

            var batchEmbeddings = model.EmbedBatch(texts);
            var individualEmbeddings = texts.Select(t => model.Embed(t)).ToList();

            for (int i = 0; i < texts.Count; i++)
            {
                for (int j = 0; j < model.EmbeddingDimension; j++)
                    Assert.Equal(individualEmbeddings[i][j], batchEmbeddings[i, j], 10);
            }
        }

        /// <summary>
        /// Deterministic mock embedding model for testing base class behavior
        /// without ONNX dependencies. Uses hash-based vector generation.
        /// </summary>
        private sealed class MockEmbeddingModel : AiDotNet.RetrievalAugmentedGeneration.Embeddings.EmbeddingModelBase<double>
        {
            private readonly int _dimension;
            public override int EmbeddingDimension => _dimension;
            public override int MaxTokens => 512;

            public MockEmbeddingModel(int dimension) { _dimension = dimension; }

            protected override Vector<double> EmbedCore(string text)
            {
                // Use stable FNV-1a hash instead of GetHashCode (randomized per process in .NET 6+)
                uint hash = 2166136261;
                foreach (char c in text.ToLowerInvariant())
                {
                    hash ^= c;
                    hash *= 16777619;
                }

                var values = new double[_dimension];
                for (int i = 0; i < _dimension; i++)
                    values[i] = Math.Sin(hash * 0.0001 + i * 0.1) * 0.5;

                var vec = new Vector<double>(values);
                return vec.Normalize();
            }

            protected override void Dispose(bool disposing) => base.Dispose(disposing);
        }
    }
}
